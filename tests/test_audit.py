from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.models.extraction import Extraction, ExtractionStatus
from app.repositories.extractions import create_extraction, get_extraction


@pytest.mark.asyncio
async def test_audit_persists_extraction() -> None:
    db = AsyncMock()
    db.add = MagicMock()
    extraction = await create_extraction(
        db,
        document_id=uuid4(),
        doc_type="invoice",
        status=ExtractionStatus.COMPLETED,
        extracted_data={"vendor_name": "Acme Corp"},
        confidence_map={"vendor_name": 0.85},
        warnings=[],
        raw_tool_output={"vendor_name": "Acme Corp"},
        model_used="test-model",
        input_tokens=100,
        output_tokens=50,
        extraction_duration_ms=25,
        source_pages=[0],
        strategy="full",
    )

    db.add.assert_called_once_with(extraction)
    db.flush.assert_awaited_once()
    db.refresh.assert_awaited_once_with(extraction)
    assert extraction.status == ExtractionStatus.COMPLETED
    assert extraction.confidence_map == {"vendor_name": 0.85}


@pytest.mark.asyncio
async def test_audit_get_returns_extraction() -> None:
    expected = Extraction(
        document_id=uuid4(),
        doc_type="invoice",
        status=ExtractionStatus.COMPLETED,
        strategy="full",
        model_used="test-model",
        input_tokens=1,
        output_tokens=2,
        extraction_duration_ms=3,
    )
    result = MagicMock()
    result.scalar_one_or_none.return_value = expected
    db = AsyncMock()
    db.execute.return_value = result

    extraction = await get_extraction(db, expected.id)

    assert extraction == expected
