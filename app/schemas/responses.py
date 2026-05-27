from typing import Any

from pydantic import BaseModel


class SchemaInfoResponse(BaseModel):
    name: str
    schema_definition: dict[str, Any]


class SchemaListResponse(BaseModel):
    schemas: list[SchemaInfoResponse]


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: str | None = None
