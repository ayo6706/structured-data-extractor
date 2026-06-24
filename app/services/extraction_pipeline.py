import time
from dataclasses import dataclass
from typing import Protocol

from app.lib.confidence import score_confidence_map
from app.schemas.requests import ExtractionStrategy
from app.services.types import StrategyExtractionResult
from app.services.validation import ValidationResult, validate_extraction


class PageRunner(Protocol):
    async def extract_from_pages(
        self,
        *,
        pages: list[str],
        doc_type: str,
        strategy: ExtractionStrategy,
    ) -> StrategyExtractionResult:
        pass


@dataclass(frozen=True)
class PipelineResult:
    doc_type: str
    validation: ValidationResult
    extraction: StrategyExtractionResult
    confidence_map: dict[str, float]
    warnings: list[str]
    duration_ms: int


class QualityPipeline:
    def __init__(self, page_runner: PageRunner) -> None:
        self.page_runner = page_runner

    async def run_resolved(
        self,
        *,
        pages: list[str],
        doc_type: str,
        strategy: ExtractionStrategy,
    ) -> PipelineResult:
        start_time = time.monotonic()
        extraction = await self.page_runner.extract_from_pages(
            pages=pages,
            doc_type=doc_type,
            strategy=strategy,
        )
        duration_ms = int((time.monotonic() - start_time) * 1000)

        validation = validate_extraction(doc_type, extraction.raw_output)
        confidence_map = score_confidence_map(
            doc_type=doc_type,
            extracted_data=validation.extracted_data,
        )

        return PipelineResult(
            doc_type=doc_type,
            validation=validation,
            extraction=extraction,
            confidence_map=confidence_map,
            warnings=validation.warnings + extraction.warnings,
            duration_ms=duration_ms,
        )
