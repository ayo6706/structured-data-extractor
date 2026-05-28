from typing import Any
from uuid import UUID

from pydantic import BaseModel


class SchemaInfoResponse(BaseModel):
    name: str
    schema_definition: dict[str, Any]


class SchemaListResponse(BaseModel):
    schemas: list[SchemaInfoResponse]


class ExtractResponse(BaseModel):
    extraction_id: UUID
    doc_type: str
    extracted_data: dict[str, Any]
    input_tokens: int
    output_tokens: int


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: str | None = None
