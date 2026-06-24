from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class SchemaInfoResponse(BaseModel):
    name: str
    schema_definition: dict[str, Any]


class SchemaListResponse(BaseModel):
    schemas: list[SchemaInfoResponse]


class ExtractResponse(BaseModel):
    extraction_id: UUID | None
    document_id: UUID | None = None
    doc_type: str
    status: str
    extracted_data: dict[str, Any] | None
    confidence_map: dict[str, float]
    warnings: list[str]
    model_used: str
    input_tokens: int
    output_tokens: int


class ExtractionResponse(BaseModel):
    extraction_id: UUID
    document_id: UUID
    doc_type: str
    status: str
    extracted_data: dict[str, Any] | None
    confidence_map: dict[str, float]
    warnings: list[str]
    model_used: str
    input_tokens: int
    output_tokens: int
    extraction_duration_ms: int


class ExtractionAuditResponse(ExtractionResponse):
    raw_tool_output: dict[str, Any] | None
    strategy: str
    source_pages: list[int] | None


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: str | None = None


class BatchItemResult(BaseModel):
    filename: str
    status: str
    document_id: UUID | None = None
    job_id: str | None = None
    extraction_id: UUID | None = None
    doc_type: str | None = None
    extracted_data: dict[str, Any] | None = None
    confidence_map: dict[str, float] | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class BatchExtractionResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[BatchItemResult]


class AsyncExtractionResponse(BaseModel):
    document_id: UUID
    job_id: str | None = None
    status: str


class DocTypeCost(BaseModel):
    count: int
    avg_extraction_ms: float
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: Decimal
    avg_cost_per_document_usd: Decimal


class UsageCost(BaseModel):
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: Decimal


class CostReportResponse(BaseModel):
    by_doc_type: dict[str, DocTypeCost]
    by_strategy: dict[str, DocTypeCost]
    by_model: dict[str, UsageCost]
    by_purpose: dict[str, UsageCost]
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost_usd: Decimal
    total_extractions: int


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    status: str
    page_count: int
    file_size_bytes: int
    uploaded_at: datetime
    extraction_ids: list[UUID]


class CorrectionResponse(BaseModel):
    correction_id: UUID
    extraction_id: UUID
    original_data: dict[str, Any] | None
    corrected_data: dict[str, Any]
    correction_diff: dict[str, Any]
    warnings: list[str]
    corrected_by: str | None
    note: str | None
    created_at: datetime
