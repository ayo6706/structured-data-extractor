from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_db
from app.main import app
from app.models.extraction import Extraction, ExtractionStatus


@pytest.fixture
def mock_db() -> AsyncMock:
    db = AsyncMock()
    app.dependency_overrides[get_db] = lambda: db

    try:
        yield db
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
async def test_get_extraction_returns_public_result(
    mock_db: AsyncMock,
) -> None:
    extraction = _extraction()
    _mock_extraction_lookup(mock_db, extraction)

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/extractions/{extraction.id}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["extraction_id"] == str(extraction.id)
    assert data["confidence_map"] == {"vendor_name": 0.85}
    assert "raw_tool_output" not in data


@pytest.mark.asyncio
async def test_get_extraction_handles_string_status_from_database(
    mock_db: AsyncMock,
) -> None:
    extraction = _extraction()
    extraction.status = "completed"
    _mock_extraction_lookup(mock_db, extraction)

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/extractions/{extraction.id}")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_get_extraction_handles_non_numeric_confidence_scores(
    mock_db: AsyncMock,
) -> None:
    extraction = _extraction()
    extraction.confidence_map = {"vendor_name": "high", "currency": 0.85}
    _mock_extraction_lookup(mock_db, extraction)

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/extractions/{extraction.id}")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["confidence_map"] == {
        "vendor_name": 0.0,
        "currency": 0.85,
    }


@pytest.mark.asyncio
async def test_get_extraction_audit_returns_raw_tool_output(
    mock_db: AsyncMock,
) -> None:
    extraction = _extraction()
    _mock_extraction_lookup(mock_db, extraction)

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/v1/extractions/{extraction.id}/audit"
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["raw_tool_output"] == {"vendor_name": "Acme Corp"}
    assert data["strategy"] == "full"
    assert data["source_pages"] == [0]


@pytest.mark.asyncio
async def test_get_extraction_returns_404(mock_db: AsyncMock) -> None:
    _mock_extraction_lookup(mock_db, None)

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/extractions/{uuid4()}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "Extraction not found"


def _mock_extraction_lookup(
    db: AsyncMock,
    extraction: Extraction | None,
) -> None:
    result = MagicMock()
    result.scalar_one_or_none.return_value = extraction
    db.execute.return_value = result


def _extraction() -> Extraction:
    return Extraction(
        document_id=uuid4(),
        doc_type="invoice",
        status=ExtractionStatus.COMPLETED,
        extracted_data={"vendor_name": "Acme Corp"},
        confidence_map={"vendor_name": 0.85},
        warnings=[],
        raw_tool_output={"vendor_name": "Acme Corp"},
        strategy="full",
        source_pages=[0],
        model_used="test-model",
        input_tokens=100,
        output_tokens=50,
        extraction_duration_ms=25,
    )
