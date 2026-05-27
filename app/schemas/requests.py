from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.registry import SchemaRegistry


class ExtractRequest(BaseModel):
    """
    Usually handled via Form data for file uploads, but if JSON is needed:
    doc_type: str | None
    strategy: "full" | "page_by_page" | "smart" | None
    """

    doc_type: str | None = Field(
        None,
        description="Optional registered document type to skip classification",
    )
    strategy: Literal["full", "page_by_page", "smart"] | None = Field(
        None, description="Optional extraction strategy"
    )

    @field_validator("doc_type")
    @classmethod
    def validate_doc_type(cls, value: str | None) -> str | None:
        if value is None:
            return value

        if value not in SchemaRegistry.list_types():
            raise ValueError(f"Unsupported document type: {value}")

        return value
