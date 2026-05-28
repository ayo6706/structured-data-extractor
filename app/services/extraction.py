import logging
import time
from typing import Final
from uuid import uuid4

import anyio
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

            full_text = "\n".join(parsed_pdf.pages_text)
            start_time = time.monotonic()
            try:
                raw_extracted, input_tokens, output_tokens = (
                    await self.extract_from_text(
                        text=full_text,
                        doc_type=resolved_type,
                    )
                )
            except ExtractionError as exc:
                logger.error("Extraction failed: %s", exc)
                db_extraction = Extraction(
                    document_id=doc_id,
                    doc_type=resolved_type,
                    status=ExtractionStatus.FAILED,
                    strategy="full",
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

            validation = validate_extraction(resolved_type, raw_extracted)
            confidence_map = score_confidence_map(
                doc_type=resolved_type,
                extracted_data=validation.extracted_data,
            )

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
                    warnings=validation.warnings,
                    raw_tool_output=raw_extracted,
                    model_used=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    extraction_duration_ms=duration_ms,
                    source_pages=list(range(parsed_pdf.page_count)),
                    strategy="full",
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
                warnings=validation.warnings,
                model_used=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
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
