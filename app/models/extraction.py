from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID, uuid4

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from app.models.document import Document


class ExtractionStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class Extraction(SQLModel, table=True):
    __tablename__: ClassVar[str] = "extractions"  # type: ignore
    __table_args__ = (
        CheckConstraint(
            "input_tokens >= 0",
            name="ck_extractions_input_tokens_non_negative",
        ),
        CheckConstraint(
            "output_tokens >= 0",
            name="ck_extractions_output_tokens_non_negative",
        ),
        CheckConstraint(
            "extraction_duration_ms >= 0",
            name="ck_extractions_duration_non_negative",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    document_id: UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    doc_type: str
    status: ExtractionStatus = Field(
        sa_column=Column(String, nullable=False, index=True)
    )
    extracted_data: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB)
    )
    confidence_map: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB)
    )
    warnings: list[str] | None = Field(
        default=None, sa_column=Column(ARRAY(String))
    )
    raw_tool_output: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB)
    )
    strategy: str
    source_pages: list[int] | None = Field(
        default=None, sa_column=Column(ARRAY(Integer))
    )
    model_used: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    extraction_duration_ms: int = Field(ge=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), index=True),
    )
    document: "Document" = Relationship(back_populates="extractions")
