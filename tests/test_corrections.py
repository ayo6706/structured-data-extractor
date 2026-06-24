from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.models.extraction import Extraction, ExtractionStatus
from app.schemas.requests import CorrectionRequest
from app.services.corrections import (
    CorrectionService,
    CorrectionValidationError,
    diff_extraction_data,
)


def test_diff_extraction_data_tracks_added_removed_and_changed_fields() -> None:
    diff = diff_extraction_data(
        {"vendor_name": "Acme", "currency": "USD", "obsolete": True},
        {"vendor_name": "Acme Ltd", "currency": "USD", "total": "10.00"},
    )

    assert diff == {
        "added": {"total": "10.00"},
        "removed": {"obsolete": True},
        "changed": {"vendor_name": {"from": "Acme", "to": "Acme Ltd"}},
    }


@pytest.mark.asyncio
async def test_correction_service_persists_valid_correction() -> None:
    db = AsyncMock()
    db.add = MagicMock()
    extraction = _invoice_extraction()

    correction = await CorrectionService().create(
        db,
        extraction=extraction,
        request=CorrectionRequest(
            corrected_data={
                "vendor_name": "Acme Ltd",
                "invoice_number": "INV-1",
                "invoice_date": "2026-05-29",
                "subtotal": "10.00",
                "total_amount": "10.00",
                "currency": "USD",
            },
            corrected_by="reviewer@example.com",
            note="Corrected legal name",
        ),
    )

    assert correction.extraction_id == extraction.id
    assert correction.original_data == extraction.extracted_data
    assert correction.corrected_data["vendor_name"] == "Acme Ltd"
    assert correction.correction_diff["changed"]["vendor_name"] == {
        "from": "Acme",
        "to": "Acme Ltd",
    }
    assert correction.corrected_by == "reviewer@example.com"
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_correction_service_rejects_unusable_correction() -> None:
    db = AsyncMock()
    db.add = MagicMock()

    with pytest.raises(CorrectionValidationError) as exc_info:
        await CorrectionService().create(
            db,
            extraction=_invoice_extraction(),
            request=CorrectionRequest(corrected_data={"unknown": "value"}),
        )

    assert exc_info.value.warnings
    db.add.assert_not_called()
    db.commit.assert_not_called()


def _invoice_extraction() -> Extraction:
    return Extraction(
        id=uuid4(),
        document_id=uuid4(),
        doc_type="invoice",
        status=ExtractionStatus.COMPLETED,
        extracted_data={
            "vendor_name": "Acme",
            "invoice_number": "INV-1",
            "invoice_date": "2026-05-29",
            "subtotal": "10.00",
            "total_amount": "10.00",
            "currency": "USD",
        },
        confidence_map={"vendor_name": 0.85},
        warnings=[],
        raw_tool_output={"vendor_name": "Acme"},
        strategy="full",
        source_pages=[0],
        model_used="test-model",
        input_tokens=10,
        output_tokens=5,
        extraction_duration_ms=20,
    )
