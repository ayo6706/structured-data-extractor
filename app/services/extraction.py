import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Final
from uuid import uuid4

import anyio
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import llm_settings
from app.core.exceptions import (
    ExtractionError,
    UnsupportedDocumentTypeError,
)
from app.infrastructure.storage import StorageBackend
from app.integrations.llm.client import LLMClient, LLMClientError
from app.lib.confidence import score_confidence_map
from app.lib.pdf import parse_pdf
from app.models.document import Document, DocumentStatus
from app.models.extraction import Extraction, ExtractionStatus
from app.repositories.extractions import create_extraction
from app.schemas.requests import ExtractionStrategy
from app.schemas.registry import SchemaRegistry
from app.schemas.responses import ExtractResponse
from app.services.classifier import ClassifierService
from app.services.validation import validate_extraction

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT: Final[str] = (
    "You are a document data extraction assistant. "
    "Your job is to extract structured data from the document text. "
    "You must call the extraction tool. Do not provide prose explanations."
)

RETRY_USER_PROMPT: Final[str] = (
    "You must respond ONLY by calling the specified tool. "
    "Do not output prose or conversational replies."
)

MONETARY_KEYWORDS: Final[tuple[str, ...]] = (
    "total",
    "amount",
    "subtotal",
    "net",
    "gross",
    "due",
    "$",
    "ngn",
    "usd",
)


@dataclass(frozen=True)
class StrategyExtractionResult:
    raw_output: dict[str, Any]
    input_tokens: int
    output_tokens: int
    strategy: ExtractionStrategy
    source_pages: list[int]
    warnings: list[str] = field(default_factory=list)
    field_source_pages: dict[str, list[int]] | None = None

    @property
    def audit_output(self) -> dict[str, Any]:
        if self.field_source_pages is None:
            return self.raw_output

        return {
            "merged_output": self.raw_output,
            "field_source_pages": self.field_source_pages,
        }


class ExtractionService:
    def __init__(
        self,
        *,
        classifier: ClassifierService,
        llm_client: LLMClient,
        storage: StorageBackend,
        model: str = llm_settings.EXTRACTION_MODEL,
        max_retries: int = llm_settings.MAX_RETRIES,
    ) -> None:
        self.classifier = classifier
        self.llm_client = llm_client
        self.storage = storage
        self.model = model
        self.max_retries = max_retries

    async def extract_upload(
        self,
        *,
        content: bytes,
        filename: str,
        doc_type: str | None,
        db: AsyncSession,
        strategy: ExtractionStrategy | str | None = None,
    ) -> ExtractResponse:
        doc_id = uuid4()
        db_document = None
        file_path = None
        document_committed = False
        failure_record_committed = False

        try:
            if doc_type is not None:
                self._validate_doc_type(doc_type)

            parsed_pdf = await anyio.to_thread.run_sync(
                parse_pdf, content, filename
            )

            file_path = await self.storage.save(
                file_id=str(doc_id),
                content=content,
                filename=filename,
            )

            db_document = Document(
                id=doc_id,
                filename=filename,
                file_path=file_path,
                file_size_bytes=len(content),
                page_count=parsed_pdf.page_count,
                status=DocumentStatus.UPLOADED,
            )
            db.add(db_document)
            await db.commit()
            document_committed = True
            await db.refresh(db_document)

            db_document.status = DocumentStatus.PROCESSING
            db.add(db_document)
            await db.commit()

            resolved_type = await self.resolve_doc_type(
                first_page_text=parsed_pdf.pages_text[0]
                if parsed_pdf.pages_text
                else "",
                doc_type=doc_type,
            )

            resolved_strategy = self._resolve_strategy(
                strategy=strategy,
            )
            start_time = time.monotonic()
            try:
                extraction_result = await self.extract_from_pages(
                    pages=parsed_pdf.pages_text,
                    doc_type=resolved_type,
                    strategy=resolved_strategy,
                )
            except ExtractionError as exc:
                logger.error("Extraction failed: %s", exc)
                db_extraction = Extraction(
                    document_id=doc_id,
                    doc_type=resolved_type,
                    status=ExtractionStatus.FAILED,
                    strategy=resolved_strategy.value,
                    model_used=self.model,
                    input_tokens=exc.input_tokens,
                    output_tokens=exc.output_tokens,
                    extraction_duration_ms=int(
                        (time.monotonic() - start_time) * 1000
                    ),
                )
                db.add(db_extraction)
                failure_record_committed = await self._mark_document_failed(
                    db, db_document
                )
                raise

            duration_ms = int((time.monotonic() - start_time) * 1000)

            validation = validate_extraction(
                resolved_type, extraction_result.raw_output
            )
            confidence_map = score_confidence_map(
                doc_type=resolved_type,
                extracted_data=validation.extracted_data,
            )
            warnings = validation.warnings + extraction_result.warnings

            db_document.status = (
                DocumentStatus.FAILED
                if validation.status == ExtractionStatus.FAILED
                else DocumentStatus.COMPLETED
            )
            db.add(db_document)
            await db.commit()

            db_extraction = None
            try:
                db_extraction = await create_extraction(
                    db,
                    document_id=doc_id,
                    doc_type=resolved_type,
                    status=validation.status,
                    extracted_data=validation.extracted_data,
                    confidence_map=confidence_map,
                    warnings=warnings,
                    raw_tool_output=extraction_result.audit_output,
                    model_used=self.model,
                    input_tokens=extraction_result.input_tokens,
                    output_tokens=extraction_result.output_tokens,
                    extraction_duration_ms=duration_ms,
                    source_pages=extraction_result.source_pages,
                    strategy=extraction_result.strategy.value,
                )
                await db.commit()
                await db.refresh(db_extraction)
            except Exception as exc:
                await db.rollback()
                db_extraction = None
                logger.error("Failed to persist extraction audit: %s", exc)

            return ExtractResponse(
                extraction_id=db_extraction.id
                if db_extraction is not None
                else None,
                doc_type=resolved_type,
                status=validation.status.value,
                extracted_data=validation.extracted_data,
                confidence_map=confidence_map,
                warnings=warnings,
                model_used=self.model,
                input_tokens=extraction_result.input_tokens,
                output_tokens=extraction_result.output_tokens,
            )
        except Exception:
            if failure_record_committed:
                raise

            await db.rollback()
            if db_document is not None and document_committed:
                await self._mark_document_failed(db, db_document)
            elif file_path is not None:
                await self._delete_orphaned_upload(file_path)
            raise

    async def resolve_doc_type(
        self,
        *,
        first_page_text: str,
        doc_type: str | None,
    ) -> str:
        if doc_type is not None:
            self._validate_doc_type(doc_type)
            return doc_type

        resolved_type = await self.classifier.classify(first_page_text)
        if resolved_type == "unknown":
            raise UnsupportedDocumentTypeError(resolved_type)

        return resolved_type

    async def extract_from_text(
        self, *, text: str, doc_type: str
    ) -> tuple[dict, int, int]:
        tool = SchemaRegistry.get_tool(doc_type)
        tool_name = f"extract_{doc_type}"

        total_input_tokens = 0
        total_output_tokens = 0
        attempts = 0
        extra_messages: list[dict[str, str]] = []

        while attempts <= self.max_retries:
            attempts += 1
            try:
                result = await self.llm_client.call_tool(
                    model=self.model,
                    system_prompt=EXTRACTION_SYSTEM_PROMPT,
                    user_content=f"Document text:\n{text}",
                    tool=tool,
                    tool_name=tool_name,
                    extra_messages=extra_messages or None,
                )
            except LLMClientError as exc:
                raise ExtractionError(
                    message="LLM tool-call extraction failed",
                    attempts=attempts,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    original_exception=exc,
                ) from exc

            if result is None:
                logger.warning(
                    "LLM did not return a tool call on attempt %d",
                    attempts,
                )
                if attempts <= self.max_retries:
                    extra_messages.append(
                        {"role": "assistant", "content": ""}
                    )
                    extra_messages.append(
                        {"role": "user", "content": RETRY_USER_PROMPT}
                    )
                    continue
                break

            total_input_tokens += result.input_tokens
            total_output_tokens += result.output_tokens
            return result.arguments, total_input_tokens, total_output_tokens

        raise ExtractionError(
            message=(
                f"Failed to extract structured data for '{doc_type}' "
                f"after {attempts} attempts. LLM failed to call tool."
            ),
            attempts=attempts,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

    async def extract_from_pages(
        self,
        *,
        pages: list[str],
        doc_type: str,
        strategy: ExtractionStrategy,
    ) -> StrategyExtractionResult:
        if strategy == ExtractionStrategy.FULL:
            return await self._extract_full_document(
                pages=pages, doc_type=doc_type
            )

        if strategy == ExtractionStrategy.SMART:
            return await self._extract_smart(pages=pages, doc_type=doc_type)

        return await self._extract_page_by_page(
            pages=pages, doc_type=doc_type
        )

    async def _extract_full_document(
        self, *, pages: list[str], doc_type: str
    ) -> StrategyExtractionResult:
        raw_output, input_tokens, output_tokens = await self.extract_from_text(
            text=self._join_pages(enumerate(pages)),
            doc_type=doc_type,
        )
        return StrategyExtractionResult(
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            strategy=ExtractionStrategy.FULL,
            source_pages=list(range(len(pages))),
        )

    async def _extract_smart(
        self, *, pages: list[str], doc_type: str
    ) -> StrategyExtractionResult:
        selected_pages = self._select_pages(pages)
        raw_output, input_tokens, output_tokens = await self.extract_from_text(
            text=self._join_pages(selected_pages),
            doc_type=doc_type,
        )
        return StrategyExtractionResult(
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            strategy=ExtractionStrategy.SMART,
            source_pages=[page_index for page_index, _ in selected_pages],
        )

    async def _extract_page_by_page(
        self, *, pages: list[str], doc_type: str
    ) -> StrategyExtractionResult:
        per_page: list[tuple[dict[str, Any], int]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        failures = 0
        warnings = []

        for page_index, page_text in enumerate(pages):
            if not page_text.strip():
                continue

            try:
                raw_output, input_tokens, output_tokens = (
                    await self.extract_from_text(
                        text=self._format_page(page_index, page_text),
                        doc_type=doc_type,
                    )
                )
            except ExtractionError as exc:
                failures += 1
                total_input_tokens += exc.input_tokens
                total_output_tokens += exc.output_tokens
                warnings.append(
                    f"page {page_index}: extraction failed and was skipped"
                )
                logger.warning(
                    "Page %d extraction failed and was skipped: %s",
                    page_index,
                    exc,
                )
                continue

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            if self._has_useful_extraction(raw_output):
                per_page.append((raw_output, page_index))

        if failures and not per_page:
            raise ExtractionError(
                message="All page-level extraction attempts failed",
                attempts=failures,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            )

        merged, field_source_pages = self._merge_results(
            per_page=per_page,
            doc_type=doc_type,
        )
        source_pages = sorted(
            {
                page_index
                for pages_for_field in field_source_pages.values()
                for page_index in pages_for_field
            }
        )

        return StrategyExtractionResult(
            raw_output=merged,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            strategy=ExtractionStrategy.PAGE_BY_PAGE,
            source_pages=source_pages,
            warnings=warnings,
            field_source_pages=field_source_pages,
        )

    @staticmethod
    def _resolve_strategy(
        *,
        strategy: ExtractionStrategy | str | None,
    ) -> ExtractionStrategy:
        if strategy is not None:
            return ExtractionStrategy(strategy)

        return ExtractionStrategy.FULL

    @staticmethod
    def _select_pages(pages: list[str]) -> list[tuple[int, str]]:
        if not pages:
            return []

        selected = {0, len(pages) - 1}
        for page_index, page_text in enumerate(pages):
            normalized = page_text.lower()
            if any(keyword in normalized for keyword in MONETARY_KEYWORDS):
                selected.add(page_index)

        return [
            (page_index, pages[page_index])
            for page_index in sorted(selected)
        ]

    def _merge_results(
        self,
        *,
        per_page: list[tuple[dict[str, Any], int]],
        doc_type: str,
    ) -> tuple[dict[str, Any], dict[str, list[int]]]:
        schema = SchemaRegistry.get_schema(doc_type)
        merged: dict[str, Any] = {}
        field_source_pages: dict[str, list[int]] = {}

        for page_output, page_index in per_page:
            for field_name in schema.model_fields:
                if field_name not in page_output:
                    continue

                candidate = page_output[field_name]
                if not self._is_useful_value(candidate):
                    continue

                if field_name not in merged:
                    merged[field_name] = candidate
                    field_source_pages[field_name] = [page_index]
                    continue

                current = merged[field_name]
                if isinstance(current, list) and isinstance(candidate, list):
                    additions = self._unique_list_items(current, candidate)
                    if additions:
                        current.extend(additions)
                        self._append_source_page(
                            field_source_pages[field_name], page_index
                        )
                    continue

                if self._is_more_complete(candidate, current):
                    merged[field_name] = candidate
                    field_source_pages[field_name] = [page_index]

        return merged, field_source_pages

    @classmethod
    def _has_useful_extraction(cls, raw_output: dict[str, Any]) -> bool:
        return any(cls._is_useful_value(value) for value in raw_output.values())

    @classmethod
    def _is_useful_value(cls, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, list | tuple | set | dict):
            return bool(value)

        return True

    @classmethod
    def _is_more_complete(cls, candidate: Any, current: Any) -> bool:
        if not cls._is_useful_value(current):
            return True
        if isinstance(candidate, str) and isinstance(current, str):
            return len(candidate.strip()) > len(current.strip())

        return False

    @classmethod
    def _unique_list_items(
        cls,
        current: list[Any],
        candidate: list[Any],
    ) -> list[Any]:
        seen = {cls._value_key(item) for item in current}
        additions = []
        for item in candidate:
            item_key = cls._value_key(item)
            if item_key in seen:
                continue
            seen.add(item_key)
            additions.append(item)

        return additions

    @staticmethod
    def _append_source_page(source_pages: list[int], page_index: int) -> None:
        if page_index not in source_pages:
            source_pages.append(page_index)

    @staticmethod
    def _value_key(value: Any) -> str:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        return repr(value)

    @classmethod
    def _join_pages(cls, pages: Iterable[tuple[int, str]]) -> str:
        return "\n\n".join(
            cls._format_page(page_index, page_text)
            for page_index, page_text in pages
        )

    @staticmethod
    def _format_page(page_index: int, page_text: str) -> str:
        return f"Page {page_index}:\n{page_text}"

    @staticmethod
    def _validate_doc_type(doc_type: str) -> None:
        if doc_type not in SchemaRegistry.list_types():
            raise UnsupportedDocumentTypeError(doc_type)

    @staticmethod
    async def _mark_document_failed(
        db: AsyncSession, document: Document
    ) -> bool:
        try:
            document.status = DocumentStatus.FAILED
            db.add(document)
            await db.commit()
            return True
        except Exception as exc:
            logger.error("Failed to set document status to FAILED: %s", exc)
            return False

    async def _delete_orphaned_upload(self, file_path: str) -> None:
        try:
            await self.storage.delete(file_path)
        except Exception as exc:
            logger.error(
                "Failed to delete orphaned upload %s: %s", file_path, exc
            )
