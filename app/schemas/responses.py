from typing import Any
from uuid import UUID

from pydantic import BaseModel


class SchemaInfoResponse(BaseModel):
    name: str
    schema_definition: dict[str, Any]


class SchemaListResponse(BaseModel):
    schemas: list[SchemaInfoResponse]


class ExtractResponse(BaseModel):
    extraction_id: UUID | None
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
