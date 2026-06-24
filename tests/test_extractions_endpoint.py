from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import (
    get_db,
    get_document_service,
)
from app.core.exceptions import (
    CorrectionValidationError,
    ExtractionNotFoundError,
)
from app.main import app
from app.models.correction import ExtractionCorrection
from app.models.extraction import Extraction, ExtractionStatus
from app.schemas.responses import (
    CorrectionResponse,
    ExtractionAuditResponse,
    ExtractionResponse,
)


@pytest.fixture
def mock_deps() -> tuple[AsyncMock, AsyncMock]:
    db = AsyncMock()
    document_service = AsyncMock()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_document_service] = lambda: document_service

    try:
        yield db, document_service
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_document_service, None)


@pytest.mark.asyncio
async def test_get_extraction_returns_public_result(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    _, document_service = mock_deps
    extraction = _extraction()
    document_service.get_extraction.return_value = _extraction_response(
        extraction
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/v1/documents/extractions/{extraction.id}"
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["extraction_id"] == str(extraction.id)
    assert data["confidence_map"] == {"vendor_name": 0.85}
    assert "raw_tool_output" not in data


@pytest.mark.asyncio
async def test_get_extraction_handles_string_status_from_database(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    _, document_service = mock_deps
    extraction = _extraction()
    response_data = _extraction_response(extraction)
    response_data.status = "completed"
    document_service.get_extraction.return_value = response_data

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/v1/documents/extractions/{extraction.id}"
        )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_get_extraction_handles_non_numeric_confidence_scores(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    _, document_service = mock_deps
    extraction = _extraction()
    response_data = _extraction_response(extraction)
    response_data.confidence_map = {"vendor_name": 0.0, "currency": 0.85}
    document_service.get_extraction.return_value = response_data

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/v1/documents/extractions/{extraction.id}"
        )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["confidence_map"] == {
        "vendor_name": 0.0,
        "currency": 0.85,
    }


@pytest.mark.asyncio
async def test_get_extraction_audit_returns_raw_tool_output(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    _, document_service = mock_deps
    extraction = _extraction()
    document_service.get_extraction_audit.return_value = (
        _extraction_audit_response(extraction)
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/v1/documents/extractions/{extraction.id}/audit"
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["raw_tool_output"] == {"vendor_name": "Acme Corp"}
    assert data["strategy"] == "full"
    assert data["source_pages"] == [0]


@pytest.mark.asyncio
async def test_get_extraction_returns_404(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    _, document_service = mock_deps
    extraction_id = uuid4()
    document_service.get_extraction.side_effect = ExtractionNotFoundError(
        extraction_id
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/v1/documents/extractions/{extraction_id}"
        )

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "Extraction not found"


@pytest.mark.asyncio
async def test_correct_extraction_returns_correction_response(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    _, document_service = mock_deps
    extraction = _extraction()
    correction = ExtractionCorrection(
        extraction_id=extraction.id,
        original_data=extraction.extracted_data,
        corrected_data={"vendor_name": "Acme Ltd"},
        correction_diff={
            "added": {},
            "removed": {},
            "changed": {"vendor_name": {"from": "Acme Corp", "to": "Acme Ltd"}},
        },
        warnings=[],
        corrected_by="reviewer@example.com",
        note="Corrected vendor name",
    )

    document_service.correct_extraction.return_value = CorrectionResponse(
        correction_id=correction.id,
        extraction_id=correction.extraction_id,
        original_data=correction.original_data,
        corrected_data=correction.corrected_data,
        correction_diff=correction.correction_diff,
        warnings=[],
        corrected_by=correction.corrected_by,
        note=correction.note,
        created_at=correction.created_at,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            f"/api/v1/documents/extractions/{extraction.id}/correct",
            json={
                "corrected_data": {"vendor_name": "Acme Ltd"},
                "corrected_by": "reviewer@example.com",
                "note": "Corrected vendor name",
            },
        )

    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["extraction_id"] == str(extraction.id)
    assert data["corrected_data"] == {"vendor_name": "Acme Ltd"}
    assert data["correction_diff"]["changed"]["vendor_name"] == {
        "from": "Acme Corp",
        "to": "Acme Ltd",
    }


@pytest.mark.asyncio
async def test_correct_extraction_returns_validation_warnings(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    _, document_service = mock_deps
    extraction = _extraction()

    document_service.correct_extraction.side_effect = CorrectionValidationError(
        ["invoice_number: Field required"]
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            f"/api/v1/documents/extractions/{extraction.id}/correct",
            json={"corrected_data": {"unknown": "value"}},
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert response.json()["detail"] == {
        "warnings": ["invoice_number: Field required"]
    }


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


def _extraction_response(extraction: Extraction) -> ExtractionResponse:
    return ExtractionResponse(
        extraction_id=extraction.id,
        document_id=extraction.document_id,
        doc_type=extraction.doc_type,
        status=extraction.status.value,
        extracted_data=extraction.extracted_data,
        confidence_map={"vendor_name": 0.85},
        warnings=[],
        model_used=extraction.model_used,
        input_tokens=extraction.input_tokens,
        output_tokens=extraction.output_tokens,
        extraction_duration_ms=extraction.extraction_duration_ms,
    )


def _extraction_audit_response(
    extraction: Extraction,
) -> ExtractionAuditResponse:
    return ExtractionAuditResponse(
        extraction_id=extraction.id,
        document_id=extraction.document_id,
        doc_type=extraction.doc_type,
        status=extraction.status.value,
        extracted_data=extraction.extracted_data,
        confidence_map={"vendor_name": 0.85},
        warnings=[],
        model_used=extraction.model_used,
        input_tokens=extraction.input_tokens,
        output_tokens=extraction.output_tokens,
        extraction_duration_ms=extraction.extraction_duration_ms,
        raw_tool_output=extraction.raw_tool_output,
        strategy=extraction.strategy,
        source_pages=extraction.source_pages,
    )
