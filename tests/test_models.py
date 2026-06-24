from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import ARRAY, Integer, String
from sqlalchemy.dialects.postgresql import JSONB

from app.models.correction import ExtractionCorrection
from app.models.document import Document, DocumentStatus
from app.models.extraction import Extraction, ExtractionStatus
from app.models.llm_usage import LLMUsage, LLMUsagePurpose


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


def test_correction_uses_jsonb_audit_columns():
    table = ExtractionCorrection.__table__

    assert isinstance(table.c.original_data.type, JSONB)
    assert isinstance(table.c.corrected_data.type, JSONB)
    assert isinstance(table.c.correction_diff.type, JSONB)
    assert isinstance(table.c.warnings.type, ARRAY)
    assert isinstance(table.c.warnings.type.item_type, String)
    foreign_key = next(iter(table.c.extraction_id.foreign_keys))
    assert foreign_key.ondelete == "RESTRICT"


def test_model_indexes_match_initial_migration():
    assert Document.__table__.c.status.index is True
    assert Document.__table__.c.uploaded_at.index is True
    assert Extraction.__table__.c.document_id.index is True
    assert Extraction.__table__.c.status.index is True
    assert Extraction.__table__.c.created_at.index is True
    assert ExtractionCorrection.__table__.c.extraction_id.index is True
    assert ExtractionCorrection.__table__.c.created_at.index is True
    assert LLMUsage.__table__.c.document_id.index is True
    assert LLMUsage.__table__.c.extraction_id.index is True
    assert LLMUsage.__table__.c.purpose.index is True
    assert LLMUsage.__table__.c.model.index is True
    assert LLMUsage.__table__.c.created_at.index is True


def test_extraction_status_values_match_lifecycle():
    assert [status.value for status in ExtractionStatus] == [
        "pending",
        "processing",
        "partial",
        "completed",
        "failed",
    ]


def test_llm_usage_purpose_values_match_cost_report_groups():
    assert [purpose.value for purpose in LLMUsagePurpose] == [
        "classifier",
        "extraction",
    ]
