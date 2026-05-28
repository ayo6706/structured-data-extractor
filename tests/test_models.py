from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import ARRAY, Integer, String
from sqlalchemy.dialects.postgresql import JSONB

from app.models.document import Document, DocumentStatus
from app.models.extraction import Extraction, ExtractionStatus


def test_document_defaults_generate_uuid_and_uploaded_at():
    document = Document(
        filename="invoice.pdf",
        file_path="/tmp/invoice.pdf",
        file_size_bytes=1024,
        page_count=2,
    )

    assert isinstance(document.id, UUID)
    assert document.status == DocumentStatus.UPLOADED
    assert isinstance(document.uploaded_at, datetime)
    assert document.uploaded_at.tzinfo == UTC


def test_extraction_uses_postgresql_specific_column_types():
    table = Extraction.__table__

    assert isinstance(table.c.extracted_data.type, JSONB)
    assert isinstance(table.c.confidence_map.type, JSONB)
    assert isinstance(table.c.raw_tool_output.type, JSONB)
    assert isinstance(table.c.warnings.type, ARRAY)
    assert isinstance(table.c.warnings.type.item_type, String)
    assert isinstance(table.c.source_pages.type, ARRAY)
    assert isinstance(table.c.source_pages.type.item_type, Integer)


def test_model_indexes_match_initial_migration():
    assert Document.__table__.c.status.index is True
    assert Document.__table__.c.uploaded_at.index is True
    assert Extraction.__table__.c.document_id.index is True
    assert Extraction.__table__.c.status.index is True
    assert Extraction.__table__.c.created_at.index is True


def test_extraction_status_values_match_lifecycle():
    assert [status.value for status in ExtractionStatus] == [
        "pending",
        "processing",
        "partial",
        "completed",
        "failed",
    ]
