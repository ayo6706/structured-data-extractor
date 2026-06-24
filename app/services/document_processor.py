from dataclasses import dataclass

from app.core.exceptions import (
    ExtractionError,
    UnsupportedDocumentTypeError,
)
from app.schemas.registry import SchemaRegistry
from app.schemas.requests import ExtractionStrategy
from app.services.classifier import ClassifierService
from app.services.extraction_pipeline import PipelineResult, QualityPipeline
from app.services.page_extraction import PageStrategyRunner
from app.services.types import LLMUsageEvent


@dataclass(frozen=True)
class ResolvedDocumentInput:
    doc_type: str
    strategy: ExtractionStrategy
    usage_events: list[LLMUsageEvent]


@dataclass(frozen=True)
class DocumentProcessingResult:
    pipeline: PipelineResult
    classifier_usage_events: list[LLMUsageEvent]


class DocumentProcessingError(Exception):
    def __init__(
        self,
        *,
        resolved: ResolvedDocumentInput,
        original: ExtractionError,
    ) -> None:
        super().__init__(str(original))
        self.resolved = resolved
        self.original = original


class DocumentProcessor:
    def __init__(
        self,
        *,
        classifier: ClassifierService,
        page_runner: PageStrategyRunner,
    ) -> None:
        self.classifier = classifier
        self.page_runner = page_runner
        self.quality_pipeline = QualityPipeline(page_runner)

    async def process_pages(
        self,
        *,
        pages: list[str],
        page_count: int,
        doc_type: str | None,
        strategy: ExtractionStrategy | str | None = None,
    ) -> DocumentProcessingResult:
        resolved = await self._resolve_extraction_input(
            pages=pages,
            page_count=page_count,
            doc_type=doc_type,
            strategy=strategy,
        )
        try:
            pipeline_result = await self.quality_pipeline.run_resolved(
                pages=pages,
                doc_type=resolved.doc_type,
                strategy=resolved.strategy,
            )
        except ExtractionError as exc:
            raise DocumentProcessingError(
                resolved=resolved,
                original=exc,
            ) from exc
        return DocumentProcessingResult(
            pipeline=pipeline_result,
            classifier_usage_events=resolved.usage_events,
        )

    async def _resolve_extraction_input(
        self,
        *,
        pages: list[str],
        page_count: int,
        doc_type: str | None,
        strategy: ExtractionStrategy | str | None,
    ) -> ResolvedDocumentInput:
        usage_events: list[LLMUsageEvent] = []
        if doc_type is None:
            first_page_text = pages[0] if pages else ""
            classification = await self.classifier.classify(first_page_text)
            doc_type = classification.doc_type
            usage_events.append(classification.usage)
        self._validate_doc_type(doc_type)

        resolved_strategy = self.page_runner.resolve_strategy(
            strategy=strategy,
            page_count=page_count,
        )
        return ResolvedDocumentInput(
            doc_type=doc_type,
            strategy=resolved_strategy,
            usage_events=usage_events,
        )

    @staticmethod
    def _validate_doc_type(doc_type: str) -> None:
        if doc_type not in SchemaRegistry.list_types():
            raise UnsupportedDocumentTypeError(doc_type)
