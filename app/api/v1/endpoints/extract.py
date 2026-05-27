import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, UploadFile, status

from app.api.dependencies import ExtractDep
from app.core.config import get_app_settings
from app.core.exceptions import (
    ClassificationError,
    PDFParseError,
    UnsupportedDocumentTypeError,
)
from app.schemas.registry import SchemaRegistry
from app.schemas.responses import ExtractResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extract", tags=["extraction"])
PDF_CONTENT_TYPE = "application/pdf"
UPLOAD_CHUNK_SIZE = 1024 * 1024


@router.post("", response_model=ExtractResponse)
async def extract_document(
    file: UploadFile,
    extract_service: ExtractDep,
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

    content = await _read_upload_file(
        file,
        max_size=get_app_settings().MAX_UPLOAD_SIZE_BYTES,
    )

    try:
        resolved_type = await extract_service.resolve_doc_type(
            content=content,
            filename=file.filename or "uploaded.pdf",
            doc_type=doc_type,
        )
    except PDFParseError as exc:
        logger.debug(
            "PDF parsing failed for %s: %s",
            file.filename,
            exc.original_exception,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not parse PDF",
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

    return ExtractResponse(doc_type=resolved_type)


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
