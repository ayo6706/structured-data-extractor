from datetime import UTC, datetime
from enum import StrEnum
from typing import ClassVar
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    String,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlmodel import Field, SQLModel


class LLMUsagePurpose(StrEnum):
    CLASSIFIER = "classifier"
    EXTRACTION = "extraction"


class LLMUsage(SQLModel, table=True):
    __tablename__: ClassVar[str] = "llm_usages"  # type: ignore
    __table_args__ = (
        CheckConstraint(
            "input_tokens >= 0",
            name="ck_llm_usages_input_tokens_non_negative",
        ),
        CheckConstraint(
            "output_tokens >= 0",
            name="ck_llm_usages_output_tokens_non_negative",
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
    extraction_id: UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("extractions.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
    )
    purpose: LLMUsagePurpose = Field(
        sa_column=Column(String, nullable=False, index=True)
    )
    model: str = Field(sa_column=Column(String, nullable=False, index=True))
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), index=True),
    )
