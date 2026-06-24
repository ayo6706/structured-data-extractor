import logging
from collections.abc import Iterable
from typing import Any, Protocol

from pydantic import BaseModel

from app.core.exceptions import ExtractionError
from app.schemas.registry import SchemaRegistry
from app.schemas.requests import ExtractionStrategy
from app.services.types import StrategyExtractionResult

logger = logging.getLogger(__name__)

MONETARY_KEYWORDS: tuple[str, ...] = (
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


class TextExtractor(Protocol):
    async def extract(
        self,
        *,
        text: str,
        doc_type: str,
    ) -> tuple[dict, int, int]:
        pass


class PageStrategyRunner:
    def __init__(self, *, text_extractor: TextExtractor) -> None:
        self.text_extractor = text_extractor

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

        return await self._extract_page_by_page(pages=pages, doc_type=doc_type)

    async def _extract_full_document(
        self, *, pages: list[str], doc_type: str
    ) -> StrategyExtractionResult:
        (
            raw_output,
            input_tokens,
            output_tokens,
        ) = await self.text_extractor.extract(
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
        selected_pages = self.select_pages(pages)
        (
            raw_output,
            input_tokens,
            output_tokens,
        ) = await self.text_extractor.extract(
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
                (
                    raw_output,
                    input_tokens,
                    output_tokens,
                ) = await self.text_extractor.extract(
                    text=self._format_page(page_index, page_text),
                    doc_type=doc_type,
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
    def resolve_strategy(
        *,
        strategy: ExtractionStrategy | str | None,
        page_count: int,
    ) -> ExtractionStrategy:
        if strategy is not None:
            return ExtractionStrategy(strategy)

        if page_count > 10:
            return ExtractionStrategy.SMART

        return ExtractionStrategy.FULL

    @staticmethod
    def select_pages(pages: list[str]) -> list[tuple[int, str]]:
        if not pages:
            return []

        selected = {0, len(pages) - 1}
        for page_index, page_text in enumerate(pages):
            normalized = page_text.lower()
            if any(keyword in normalized for keyword in MONETARY_KEYWORDS):
                selected.add(page_index)

        return [
            (page_index, pages[page_index]) for page_index in sorted(selected)
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
                    self._append_source_page(
                        field_source_pages[field_name], page_index
                    )

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


PageExtractionService = PageStrategyRunner
TextExtractionService = TextExtractor
