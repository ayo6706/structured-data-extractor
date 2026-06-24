import time
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ExtractionError
from app.models.document import Document
from app.models.extraction import Extraction, ExtractionStatus
from app.models.llm_usage import LLMUsagePurpose
from app.repositories.extractions import ExtractionRepository
from app.repositories.llm_usage import LLMUsageRepository
from app.repositories.transactions import refresh
from app.schemas.responses import ExtractResponse
from app.services.extraction_pipeline import PipelineResult
from app.services.types import LLMUsageEvent


class AuditRecorder:
    def __init__(
        self,
        *,
        db: AsyncSession,
        extractions: ExtractionRepository,
        llm_usage: LLMUsageRepository,
        model: str,
    ) -> None:
        self.db = db
        self.extractions = extractions
        self.llm_usage = llm_usage
        self.model = model

    async def record_failure(
        self,
        *,
        document: Document,
        doc_type: str,
        strategy: str,
        started_at: float,
        exc: ExtractionError,
        classifier_usage_events: list[LLMUsageEvent],
    ) -> None:
        extraction = await self.extractions.create(
            Extraction(
                document_id=document.id,
                doc_type=doc_type,
                status=ExtractionStatus.FAILED,
                strategy=strategy,
                model_used=self.model,
                input_tokens=exc.input_tokens,
                output_tokens=exc.output_tokens,
                extraction_duration_ms=int(
                    (time.monotonic() - started_at) * 1000
                ),
            ),
        )
        await self._add_usage_events(
            document_id=document.id,
            extraction_id=extraction.id,
            classifier_usage_events=classifier_usage_events,
            extraction_input_tokens=exc.input_tokens,
            extraction_output_tokens=exc.output_tokens,
        )

    async def record_success(
        self,
        *,
        document: Document,
        result: PipelineResult,
        classifier_usage_events: list[LLMUsageEvent],
    ) -> ExtractResponse:
        db_extraction = await self.extractions.create_from_fields(
            document_id=document.id,
            doc_type=result.doc_type,
            status=result.validation.status,
            extracted_data=result.validation.extracted_data,
            confidence_map=result.confidence_map,
            warnings=result.warnings,
            raw_tool_output=result.extraction.audit_output,
            model_used=self.model,
            input_tokens=result.extraction.input_tokens,
            output_tokens=result.extraction.output_tokens,
            extraction_duration_ms=result.duration_ms,
            source_pages=result.extraction.source_pages,
            strategy=result.extraction.strategy.value,
        )
        await self._add_usage_events(
            document_id=document.id,
            extraction_id=db_extraction.id,
            classifier_usage_events=classifier_usage_events,
            extraction_input_tokens=result.extraction.input_tokens,
            extraction_output_tokens=result.extraction.output_tokens,
        )
        await refresh(self.db, db_extraction)

        return self.to_response(
            extraction_id=db_extraction.id,
            document_id=document.id,
            result=result,
        )

    def to_response(
        self,
        *,
        extraction_id: UUID,
        document_id: UUID,
        result: PipelineResult,
    ) -> ExtractResponse:
        return ExtractResponse(
            extraction_id=extraction_id,
            document_id=document_id,
            doc_type=result.doc_type,
            status=result.validation.status.value,
            extracted_data=result.validation.extracted_data,
            confidence_map=result.confidence_map,
            warnings=result.warnings,
            model_used=self.model,
            input_tokens=result.extraction.input_tokens,
            output_tokens=result.extraction.output_tokens,
        )

    async def _add_usage_events(
        self,
        *,
        document_id: UUID,
        extraction_id: UUID,
        classifier_usage_events: list[LLMUsageEvent],
        extraction_input_tokens: int,
        extraction_output_tokens: int,
    ) -> None:
        for usage in classifier_usage_events:
            await self.llm_usage.create(
                document_id=document_id,
                extraction_id=extraction_id,
                purpose=usage.purpose,
                model=usage.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )

        await self.llm_usage.create(
            document_id=document_id,
            extraction_id=extraction_id,
            purpose=LLMUsagePurpose.EXTRACTION,
            model=self.model,
            input_tokens=extraction_input_tokens,
            output_tokens=extraction_output_tokens,
        )


ExtractionAuditService = AuditRecorder
