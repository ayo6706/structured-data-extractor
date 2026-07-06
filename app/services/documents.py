import asyncio
import logging
import time
from datetime import datetime
from typing import Any
from uuid import UUID

from arq.connections import ArqRedis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    DocumentNotFoundError,
    ExtractionNotFoundError,
    InvalidUploadError,
)
from app.infrastructure.storage import StorageBackend
from app.jobs.extraction_queue import (
    REDIS_FALLBACK_WARNING,
    enqueue_extraction,
)
from app.models.document import Document, DocumentStatus
from app.models.extraction import Extraction, ExtractionStatus
from app.repositories.documents import DocumentRepository
from app.repositories.extractions import ExtractionRepository
from app.repositories.llm_usage import LLMUsageRepository
from app.schemas.requests import CorrectionRequest, ExtractionStrategy
from app.schemas.responses import (
    AsyncExtractionResponse,
    BatchExtractionResponse,
    BatchItemResult,
    CorrectionResponse,
    CostReportResponse,
    DocumentResponse,
    ExtractionAuditResponse,
    ExtractionResponse,
    ExtractResponse,
)
from app.services.audit import AuditRecorder
from app.services.corrections import CorrectionService
from app.services.cost_report import CostReportService, PriceResolver
from app.services.document_intake import DocumentIntakeService
from app.services.document_processor import (
    DocumentProcessingError,
    DocumentProcessingResult,
    DocumentProcessor,
)

logger = logging.getLogger(__name__)
BATCH_CONCURRENCY_LIMIT = 5


class DocumentService:
    def __init__(
        self,
        *,
        db: AsyncSession,
        processor: DocumentProcessor,
        storage: StorageBackend,
        model: str,
        arq_pool: ArqRedis | None,
        session_factory: Any,
        price_for_model: PriceResolver,
    ) -> None:
        self.db = db
        self.processor = processor
        self.storage = storage
        self.model = model
        self.arq_pool = arq_pool
        self.session_factory = session_factory
        self.price_for_model = price_for_model
        self.documents = DocumentRepository(db)
        self.extractions = ExtractionRepository(db)
        self.intake = DocumentIntakeService(storage=storage)
        self.audit = AuditRecorder(
            db=db,
            extractions=ExtractionRepository(db),
            llm_usage=LLMUsageRepository(db),
            model=model,
        )
        self.batch_concurrency_limit = BATCH_CONCURRENCY_LIMIT
        self.corrections = CorrectionService(db)
        self.cost_report = CostReportService(
            db=db,
            price_for_model=price_for_model,
        )

    def _for_session(self, db: AsyncSession) -> "DocumentService":
        return DocumentService(
            db=db,
            processor=self.processor,
            storage=self.storage,
            model=self.model,
            arq_pool=self.arq_pool,
            session_factory=self.session_factory,
            price_for_model=self.price_for_model,
        )

    async def get_document(
        self,
        document_id: UUID,
    ) -> DocumentResponse:
        document = await self.documents.get_with_extractions(document_id)
        if document is None:
            raise DocumentNotFoundError(document_id)

        return DocumentResponse(
            id=document.id,
            filename=document.filename,
            status=document.status,
            page_count=document.page_count,
            file_size_bytes=document.file_size_bytes,
            uploaded_at=document.uploaded_at,
            extraction_ids=[
                extraction.id for extraction in document.extractions
            ],
        )

    async def extract_document(
        self,
        *,
        content: bytes | None,
        filename: str | None,
        document_id: UUID | None,
        doc_type: str | None,
        strategy: ExtractionStrategy | None,
        async_mode: bool,
    ) -> ExtractResponse | AsyncExtractionResponse:
        has_upload = content is not None and filename is not None
        has_document = document_id is not None
        if has_upload == has_document:
            raise InvalidUploadError(
                "Provide either a PDF file or document_id, but not both"
            )

        if has_document:
            return await self._extract_existing_document(
                document_id=document_id,
                doc_type=doc_type,
                strategy=strategy,
                async_mode=async_mode,
            )

        return await self._extract_uploaded_document(
            content=content,
            filename=filename,
            doc_type=doc_type,
            strategy=strategy,
            async_mode=async_mode,
        )

    async def _extract_uploaded_document(
        self,
        *,
        content: bytes,
        filename: str,
        doc_type: str | None,
        strategy: ExtractionStrategy | None,
        async_mode: bool,
    ) -> ExtractResponse | AsyncExtractionResponse:
        if not async_mode or self.arq_pool is None:
            result = await self._process_uploaded_inline(
                content=content,
                filename=filename,
                doc_type=doc_type,
                strategy=strategy,
            )
            if async_mode:
                result.warnings.append(REDIS_FALLBACK_WARNING)
            return result

        intake = await self.intake.prepare_document(
            content=content,
            filename=filename,
            status=DocumentStatus.PROCESSING,
        )
        try:
            await self._create_document(intake.document)
        except Exception:
            await self.db.rollback()
            await self.intake.delete_upload(intake.document.file_path)
            raise
        document = intake.document
        try:
            job = await enqueue_extraction(
                arq_pool=self.arq_pool,
                document_id=document.id,
                doc_type=doc_type,
                strategy=strategy,
            )
            return AsyncExtractionResponse(
                document_id=document.id,
                job_id=job.job_id,
                status="processing",
            )
        except Exception as exc:
            logger.warning(
                "Arq enqueue failed: %s. Falling back to inline.",
                exc,
            )
            result = await self._process_document_pages(
                document=document,
                pages=intake.pages_text,
                doc_type=doc_type,
                strategy=strategy,
            )
            result.warnings.append(REDIS_FALLBACK_WARNING)
            return result

    async def _extract_existing_document(
        self,
        *,
        document_id: UUID,
        doc_type: str | None,
        strategy: ExtractionStrategy | None,
        async_mode: bool,
    ) -> ExtractResponse | AsyncExtractionResponse:
        if async_mode and self.arq_pool is not None:
            document = await self.documents.get_by_id(document_id)
            if document is None:
                raise DocumentNotFoundError(document_id)

            job = await enqueue_extraction(
                arq_pool=self.arq_pool,
                document_id=document_id,
                doc_type=doc_type,
                strategy=strategy,
            )
            return AsyncExtractionResponse(
                document_id=document_id,
                job_id=job.job_id,
                status="processing",
            )

        result = await self._process_existing_inline(
            document_id=document_id,
            doc_type=doc_type,
            strategy=strategy,
        )
        if async_mode:
            result.warnings.append(REDIS_FALLBACK_WARNING)
        return result

    async def _process_uploaded_inline(
        self,
        *,
        content: bytes,
        filename: str,
        doc_type: str | None,
        strategy: ExtractionStrategy | None,
    ) -> ExtractResponse:
        intake = await self.intake.prepare_document(
            content=content,
            filename=filename,
            status=DocumentStatus.UPLOADED,
        )
        document = intake.document
        document_created = False
        try:
            await self._create_document(document)
            document_created = True
            await self._set_document_status(
                document,
                DocumentStatus.PROCESSING,
            )
            return await self._process_document_pages(
                document=document,
                pages=intake.pages_text,
                doc_type=doc_type,
                strategy=strategy,
            )
        except Exception:
            await self.db.rollback()
            if not document_created:
                await self.intake.delete_upload(document.file_path)
            raise

    async def _process_existing_inline(
        self,
        *,
        document_id: UUID,
        doc_type: str | None,
        strategy: ExtractionStrategy | None,
    ) -> ExtractResponse:
        document = await self.documents.get_by_id(document_id)
        if document is None:
            raise DocumentNotFoundError(document_id)

        await self._set_document_status(document, DocumentStatus.PROCESSING)
        try:
            pages = await self.intake.load_document(document)
            return await self._process_document_pages(
                document=document,
                pages=pages,
                doc_type=doc_type,
                strategy=strategy,
            )
        except Exception:
            await self.documents.set_status(document, DocumentStatus.FAILED)
            await self.db.commit()
            raise

    async def _process_document_pages(
        self,
        *,
        document: Document,
        pages: list[str],
        doc_type: str | None,
        strategy: ExtractionStrategy | None,
    ) -> ExtractResponse:
        started_at = time.monotonic()
        try:
            processing = await self.processor.process_pages(
                pages=pages,
                page_count=document.page_count,
                doc_type=doc_type,
                strategy=strategy,
            )
        except DocumentProcessingError as exc:
            await self._record_processing_failure(
                document=document,
                error=exc,
                started_at=started_at,
            )
            raise exc.original from exc

        return await self._record_processing_success(document, processing)

    async def _record_processing_success(
        self,
        document: Document,
        processing: DocumentProcessingResult,
    ) -> ExtractResponse:
        await self.documents.set_status(
            document,
            _document_status_for(processing),
        )
        response = await self.audit.record_success(
            document=document,
            result=processing.pipeline,
            classifier_usage_events=processing.classifier_usage_events,
        )
        await self.db.commit()
        return response

    async def _record_processing_failure(
        self,
        *,
        document: Document,
        error: DocumentProcessingError,
        started_at: float,
    ) -> None:
        await self.documents.set_status(document, DocumentStatus.FAILED)
        await self.audit.record_failure(
            document=document,
            doc_type=error.resolved.doc_type,
            strategy=error.resolved.strategy.value,
            started_at=started_at,
            exc=error.original,
            classifier_usage_events=error.resolved.usage_events,
        )
        await self.db.commit()

    async def _create_document(self, document: Document) -> None:
        await self.documents.create(document)
        await self.db.commit()
        await self.db.refresh(document)

    async def _set_document_status(
        self,
        document: Document,
        status: DocumentStatus,
    ) -> None:
        await self.documents.set_status(document, status)
        await self.db.commit()

    async def batch_extract(
        self,
        *,
        file_data: list[tuple[str, bytes]],
        doc_type: str | None,
        strategy: ExtractionStrategy | None,
        async_mode: bool,
    ) -> BatchExtractionResponse:
        results = await self._extract_batch_items(
            file_data=file_data,
            doc_type=doc_type,
            strategy=strategy,
            async_mode=async_mode,
        )

        return BatchExtractionResponse(
            total=len(file_data),
            succeeded=sum(
                result.status in ("completed", "partial", "processing")
                for result in results
            ),
            failed=sum(result.status == "failed" for result in results),
            results=results,
        )

    async def _extract_batch_items(
        self,
        *,
        file_data: list[tuple[str, bytes]],
        doc_type: str | None,
        strategy: ExtractionStrategy | None,
        async_mode: bool,
    ) -> list[BatchItemResult]:
        semaphore = asyncio.Semaphore(self.batch_concurrency_limit)

        async def extract_one(filename: str, content: bytes) -> BatchItemResult:
            async with semaphore:
                return await self._extract_batch_item(
                    filename=filename,
                    content=content,
                    doc_type=doc_type,
                    strategy=strategy,
                    async_mode=async_mode,
                )

        results = await asyncio.gather(
            *[
                extract_one(filename, content)
                for filename, content in file_data
            ],
            return_exceptions=True,
        )
        return [
            result
            if isinstance(result, BatchItemResult)
            else BatchItemResult(
                filename=file_data[index][0],
                status="failed",
                error=str(result),
            )
            for index, result in enumerate(results)
        ]

    async def _extract_batch_item(
        self,
        *,
        filename: str,
        content: bytes,
        doc_type: str | None,
        strategy: ExtractionStrategy | None,
        async_mode: bool,
    ) -> BatchItemResult:
        async with self.session_factory() as db:
            documents = self._for_session(db)
            try:
                result = await documents.extract_document(
                    content=content,
                    filename=filename,
                    document_id=None,
                    doc_type=doc_type,
                    strategy=strategy,
                    async_mode=async_mode,
                )
                return _to_batch_item_result(filename, result)
            except Exception as exc:
                logger.exception(
                    "Batch extraction failed for file %s", filename
                )
                return BatchItemResult(
                    filename=filename,
                    status="failed",
                    error=str(exc),
                )

    async def get_cost_report(
        self,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> CostReportResponse:
        return await self.cost_report.generate(from_date, to_date)

    async def get_extraction(
        self,
        extraction_id: UUID,
    ) -> ExtractionResponse:
        extraction = await self.require_extraction(extraction_id)
        return _to_extraction_response(extraction)

    async def get_extraction_audit(
        self,
        extraction_id: UUID,
    ) -> ExtractionAuditResponse:
        extraction = await self.require_extraction(extraction_id)
        return _to_extraction_audit_response(extraction)

    async def require_extraction(
        self,
        extraction_id: UUID,
    ) -> Extraction:
        extraction = await self.extractions.get_by_id(extraction_id)
        if extraction is None:
            raise ExtractionNotFoundError(extraction_id)

        return extraction

    async def correct_extraction(
        self,
        extraction_id: UUID,
        request: CorrectionRequest,
    ) -> CorrectionResponse:
        extraction = await self.require_extraction(extraction_id)
        return await self.corrections.correct(extraction, request)


def _to_batch_item_result(
    filename: str,
    result: ExtractResponse | AsyncExtractionResponse,
) -> BatchItemResult:
    if isinstance(result, AsyncExtractionResponse):
        return BatchItemResult(
            filename=filename,
            status=result.status,
            document_id=result.document_id,
            job_id=result.job_id,
        )

    return BatchItemResult(
        filename=filename,
        status=result.status,
        document_id=result.document_id,
        extraction_id=result.extraction_id,
        doc_type=result.doc_type,
        extracted_data=result.extracted_data,
        confidence_map=result.confidence_map,
        warnings=list(result.warnings),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def _to_extraction_response(extraction: Extraction) -> ExtractionResponse:
    return ExtractionResponse(
        extraction_id=extraction.id,
        document_id=extraction.document_id,
        doc_type=extraction.doc_type,
        status=_status_value(extraction.status),
        extracted_data=extraction.extracted_data,
        confidence_map=_float_map(extraction.confidence_map),
        warnings=extraction.warnings or [],
        model_used=extraction.model_used,
        input_tokens=extraction.input_tokens,
        output_tokens=extraction.output_tokens,
        extraction_duration_ms=extraction.extraction_duration_ms,
    )


def _document_status_for(
    processing: DocumentProcessingResult,
) -> DocumentStatus:
    if processing.pipeline.validation.status == ExtractionStatus.FAILED:
        return DocumentStatus.FAILED

    return DocumentStatus.COMPLETED


def _to_extraction_audit_response(
    extraction: Extraction,
) -> ExtractionAuditResponse:
    return ExtractionAuditResponse(
        extraction_id=extraction.id,
        document_id=extraction.document_id,
        doc_type=extraction.doc_type,
        status=_status_value(extraction.status),
        extracted_data=extraction.extracted_data,
        confidence_map=_float_map(extraction.confidence_map),
        warnings=extraction.warnings or [],
        model_used=extraction.model_used,
        input_tokens=extraction.input_tokens,
        output_tokens=extraction.output_tokens,
        extraction_duration_ms=extraction.extraction_duration_ms,
        raw_tool_output=extraction.raw_tool_output,
        strategy=extraction.strategy,
        source_pages=extraction.source_pages,
    )


def _float_map(value: dict[str, object] | None) -> dict[str, float]:
    scores = {}
    for key, score in (value or {}).items():
        try:
            scores[key] = float(score)
        except (TypeError, ValueError):
            scores[key] = 0.0

    return scores


def _status_value(status: ExtractionStatus | str) -> str:
    if isinstance(status, ExtractionStatus):
        return status.value

    return status
