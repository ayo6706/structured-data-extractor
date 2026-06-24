from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID, uuid4

from sqlalchemy import ARRAY, Column, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlmodel import Field, SQLModel


class ExtractionCorrection(SQLModel, table=True):
    __tablename__: ClassVar[str] = "extraction_corrections"  # type: ignore

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    extraction_id: UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("extractions.id", ondelete="RESTRICT"),
            nullable=False,
            index=True,
        )
    )
    original_data: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB)
    )
    corrected_data: dict[str, Any] = Field(sa_column=Column(JSONB))
    correction_diff: dict[str, Any] = Field(sa_column=Column(JSONB))
    warnings: list[str] = Field(
        default_factory=list, sa_column=Column(ARRAY(String))
    )
    corrected_by: str | None = Field(default=None, sa_column=Column(String))
    note: str | None = Field(default=None, sa_column=Column(String))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), index=True),
    )
