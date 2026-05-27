from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, Column, DateTime, String
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from app.models.extraction import Extraction


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(SQLModel, table=True):
    __tablename__: ClassVar[str] = "documents"  # type: ignore
    __table_args__ = (
        CheckConstraint(
            "file_size_bytes > 0",
            name="ck_documents_file_size_positive",
        ),
        CheckConstraint(
            "page_count > 0",
            name="ck_documents_page_count_positive",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    filename: str
    file_path: str
    file_size_bytes: int = Field(gt=0)
    page_count: int = Field(gt=0)
    status: DocumentStatus = Field(
        default=DocumentStatus.UPLOADED,
        sa_column=Column(String, nullable=False, index=True),
    )
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), index=True),
    )
    extractions: list["Extraction"] = Relationship(back_populates="document")
