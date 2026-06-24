from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import (
    DocumentServiceDep,
)
from app.api.validation import (
    UploadedDocument,
    optional_pdf_upload,
    valid_doc_type,
    valid_pdf_batch,
)
from app.schemas.requests import CorrectionRequest, ExtractionStrategy
from app.schemas.responses import (
    AsyncExtractionResponse,
    BatchExtractionResponse,
    CorrectionResponse,
    CostReportResponse,
    DocumentResponse,
    ExtractionAuditResponse,
    ExtractionResponse,
    ExtractResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])
DocTypeDep = Annotated[str | None, Depends(valid_doc_type)]
UploadDep = Annotated[UploadedDocument | None, Depends(optional_pdf_upload)]
BatchUploadDep = Annotated[list[UploadedDocument], Depends(valid_pdf_batch)]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    documents: DocumentServiceDep,
) -> DocumentResponse:
    return await documents.get_document(document_id)


@router.post(
    "/extractions",
    response_model=ExtractResponse | AsyncExtractionResponse,
)
async def extract_document(
    upload: UploadDep,
    documents: DocumentServiceDep,
    doc_type: DocTypeDep,
    document_id: Annotated[
        UUID | None,
        Query(description="Extract an already uploaded document"),
    ] = None,
    strategy: Annotated[
        ExtractionStrategy | None,
        Query(description="Optional extraction strategy"),
    ] = None,
    async_mode: Annotated[
        bool,
        Query(alias="async", description="Run extraction in background"),
    ] = False,
) -> ExtractResponse | AsyncExtractionResponse:
    return await documents.extract_document(
        content=upload.content if upload is not None else None,
        filename=upload.filename if upload is not None else None,
        document_id=document_id,
        doc_type=doc_type,
        strategy=strategy,
        async_mode=async_mode,
    )


@router.post("/extractions/batch", response_model=BatchExtractionResponse)
async def batch_extract(
    uploads: BatchUploadDep,
    documents: DocumentServiceDep,
    doc_type: DocTypeDep,
    strategy: Annotated[ExtractionStrategy | None, Query()] = None,
    async_mode: Annotated[bool, Query(alias="async")] = True,
) -> BatchExtractionResponse:
    return await documents.batch_extract(
        file_data=[(upload.filename, upload.content) for upload in uploads],
        doc_type=doc_type,
        strategy=strategy,
        async_mode=async_mode,
    )


@router.get("/extractions/cost-report", response_model=CostReportResponse)
async def get_cost_report(
    documents: DocumentServiceDep,
    from_date: Annotated[
        datetime | None,
        Query(alias="from", description="Filter from date-time"),
    ] = None,
    to_date: Annotated[
        datetime | None,
        Query(alias="to", description="Filter to date-time"),
    ] = None,
) -> CostReportResponse:
    return await documents.get_cost_report(from_date, to_date)


@router.get("/extractions/{extraction_id}", response_model=ExtractionResponse)
async def get_extraction(
    extraction_id: UUID,
    documents: DocumentServiceDep,
) -> ExtractionResponse:
    return await documents.get_extraction(extraction_id)


@router.get(
    "/extractions/{extraction_id}/audit",
    response_model=ExtractionAuditResponse,
)
async def get_extraction_audit(
    extraction_id: UUID,
    documents: DocumentServiceDep,
) -> ExtractionAuditResponse:
    return await documents.get_extraction_audit(extraction_id)


@router.post(
    "/extractions/{extraction_id}/correct",
    response_model=CorrectionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def correct_extraction(
    extraction_id: UUID,
    request: CorrectionRequest,
    documents: DocumentServiceDep,
) -> CorrectionResponse:
    return await documents.correct_extraction(extraction_id, request)
