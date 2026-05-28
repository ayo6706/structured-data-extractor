import logging
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import DbDep, ExtractionDep
from app.core.config import get_app_settings
from app.core.exceptions import (
    ClassificationError,
    ExtractionError,
    PDFParseError,
    StorageError,
    UnsupportedDocumentTypeError,
)
from app.models.extraction import Extraction, ExtractionStatus
from app.repositories.extractions import get_extraction as get_extraction_row
from app.schemas.registry import SchemaRegistry
from app.schemas.responses import (
    ExtractionAuditResponse,
    ExtractionResponse,
    ExtractResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["extractions"])
PDF_CONTENT_TYPE = "application/pdf"
UPLOAD_CHUNK_SIZE = 1024 * 1024


@router.post("/extract", response_model=ExtractResponse)
async def extract_document(
    file: UploadFile,
    db: DbDep,
    extraction_service: ExtractionDep,
    doc_type: Annotated[
        str | None,
        Query(
            description=(
                "Skip classification by providing a known document type"
            )
        ),
    ] = None,
) -> ExtractResponse:
    if not _is_pdf_upload(file):
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF uploads are supported",
        )

    filename = file.filename or "uploaded.pdf"
    content = await _read_upload_file(
        file,
        max_size=get_app_settings().MAX_UPLOAD_SIZE_BYTES,
    )

    try:
        return await extraction_service.extract_upload(
            content=content,
            filename=filename,
            doc_type=doc_type,
            db=db,
        )
    except PDFParseError as exc:
        logger.debug(
            "PDF parsing failed for %s: %s",
            filename,
            exc.original_exception,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not parse PDF",
        ) from exc
    except StorageError as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file to storage",
        ) from exc
    except ClassificationError as exc:
        logger.error("Classification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Classification service unavailable",
        ) from exc
    except UnsupportedDocumentTypeError as exc:
        if doc_type is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid doc_type '{exc.document_type}'. "
                    f"Available types: {SchemaRegistry.list_types()}"
                ),
            ) from exc

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Could not determine the document type. "
                f"Supported types: {SchemaRegistry.list_types()}"
            ),
        ) from exc
    except ExtractionError as exc:
        logger.error("Extraction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {exc}",
        ) from exc


@router.get("/extractions/{extraction_id}", response_model=ExtractionResponse)
async def get_extraction(
    extraction_id: UUID,
    db: DbDep,
) -> ExtractionResponse:
    extraction = await _get_extraction_or_404(db, extraction_id)
    return _to_response(extraction)


@router.get(
    "/extractions/{extraction_id}/audit",
    response_model=ExtractionAuditResponse,
)
async def get_extraction_audit(
    extraction_id: UUID,
    db: DbDep,
) -> ExtractionAuditResponse:
    extraction = await _get_extraction_or_404(db, extraction_id)
    return _to_audit_response(extraction)


async def _get_extraction_or_404(
    db: AsyncSession,
    extraction_id: UUID,
) -> Extraction:
    extraction = await get_extraction_row(db, extraction_id)
    if extraction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Extraction not found",
        )

    return extraction


def _to_response(extraction: Extraction) -> ExtractionResponse:
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


def _to_audit_response(extraction: Extraction) -> ExtractionAuditResponse:
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


def _float_map(value: dict[str, Any] | None) -> dict[str, float]:
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


def _is_pdf_upload(file: UploadFile) -> bool:
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    return content_type == PDF_CONTENT_TYPE or filename.endswith(".pdf")


async def _read_upload_file(
    file: UploadFile,
    *,
    max_size: int,
    chunk_size: int = UPLOAD_CHUNK_SIZE,
) -> bytes:
    chunks = []
    total_size = 0

    while chunk := await file.read(chunk_size):
        total_size += len(chunk)
        if total_size > max_size:
            await file.close()
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Uploaded file is too large",
            )
        chunks.append(chunk)

    return b"".join(chunks)
