from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_db, get_extraction_service
from app.core.exceptions import (
    ClassificationError,
    ExtractionError,
    PDFParseError,
    StorageError,
    UnsupportedDocumentTypeError,
)
from app.main import app
from app.schemas.responses import ExtractResponse


@pytest.fixture
def mock_deps() -> tuple[AsyncMock, AsyncMock]:
    db = AsyncMock()
    extraction_service = AsyncMock()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_extraction_service] = (
        lambda: extraction_service
    )

    try:
        yield db, extraction_service
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_extraction_service, None)


@pytest.mark.asyncio
async def test_extract_success(mock_deps: tuple[AsyncMock, AsyncMock]) -> None:
    _, extraction_service = mock_deps
    extraction_service.extract_upload.return_value = ExtractResponse(
        extraction_id=uuid4(),
        doc_type="contract",
        status="completed",
        extracted_data={
            "parties": [
                {"name": "Client Corp", "role": "Client"},
                {"name": "Vendor Corp", "role": "Vendor"},
            ],
            "effective_date": "2026-05-27",
            "key_obligations": ["Deliver goods"],
        },
        confidence_map={"parties": 0.85},
        warnings=[],
        model_used="test-model",
        input_tokens=100,
        output_tokens=50,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            params={"doc_type": "contract"},
            files={"file": ("test.pdf", b"%PDF-1.7", "application/pdf")},
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["doc_type"] == "contract"
    assert data["status"] == "completed"
    assert data["confidence_map"] == {"parties": 0.85}
    assert data["warnings"] == []
    assert data["model_used"] == "test-model"
    assert data["input_tokens"] == 100
    assert data["output_tokens"] == 50
    extraction_service.extract_upload.assert_called_once()


@pytest.mark.asyncio
async def test_extract_validation_failed_status(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    _, extraction_service = mock_deps
    extraction_service.extract_upload.return_value = ExtractResponse(
        extraction_id=None,
        doc_type="contract",
        status="failed",
        extracted_data=None,
        confidence_map={"parties": 0.0},
        warnings=["parties: Field required"],
        model_used="test-model",
        input_tokens=100,
        output_tokens=50,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            params={"doc_type": "contract"},
            files={"file": ("test.pdf", b"%PDF-1.7", "application/pdf")},
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "failed"
    assert data["extraction_id"] is None
    assert data["warnings"] == ["parties: Field required"]


@pytest.mark.asyncio
async def test_extract_rejects_non_pdf_upload(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    _, extraction_service = mock_deps
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("test.txt", b"text", "text/plain")},
        )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Only PDF uploads are supported"
    extraction_service.extract_upload.assert_not_called()


@pytest.mark.asyncio
async def test_extract_rejects_oversized_upload(
    mock_deps: tuple[AsyncMock, AsyncMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, extraction_service = mock_deps
    monkeypatch.setattr(
        "app.api.v1.endpoints.extractions.get_app_settings",
        lambda: SimpleNamespace(MAX_UPLOAD_SIZE_BYTES=3),
    )
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("test.pdf", b"1234", "application/pdf")},
        )

    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert response.json()["detail"] == "Uploaded file is too large"
    extraction_service.extract_upload.assert_not_called()


@pytest.mark.asyncio
async def test_extract_invalid_pdf(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    _, extraction_service = mock_deps
    extraction_service.extract_upload.side_effect = PDFParseError(
        "test.pdf", Exception("bad pdf")
    )
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("test.pdf", b"garbage", "application/pdf")},
        )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Could not parse PDF"


@pytest.mark.asyncio
async def test_extract_storage_error(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    _, extraction_service = mock_deps
    extraction_service.extract_upload.side_effect = StorageError(
        path="path", operation="save", original_exception=Exception("full")
    )
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            params={"doc_type": "contract"},
            files={"file": ("test.pdf", b"%PDF-1.7", "application/pdf")},
        )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert (
        "Failed to save uploaded file to storage" in response.json()["detail"]
    )


@pytest.mark.asyncio
async def test_extract_classification_error(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    _, extraction_service = mock_deps
    extraction_service.extract_upload.side_effect = ClassificationError(
        "LLM failed"
    )
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("test.pdf", b"%PDF-1.7", "application/pdf")},
        )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert "Classification service unavailable" in response.json()["detail"]


@pytest.mark.asyncio
async def test_extract_explicit_doc_type_invalid(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    _, extraction_service = mock_deps
    extraction_service.extract_upload.side_effect = (
        UnsupportedDocumentTypeError("banana")
    )
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            params={"doc_type": "banana"},
            files={"file": ("test.pdf", b"%PDF-1.7", "application/pdf")},
        )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid doc_type 'banana'" in response.json()["detail"]


@pytest.mark.asyncio
async def test_extract_auto_unknown(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    _, extraction_service = mock_deps
    extraction_service.extract_upload.side_effect = (
        UnsupportedDocumentTypeError("unknown")
    )
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("test.pdf", b"%PDF-1.7", "application/pdf")},
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert "Could not determine the document type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_extract_engine_error(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    _, extraction_service = mock_deps
    extraction_service.extract_upload.side_effect = ExtractionError(
        "LLM failed", attempts=3
    )
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            params={"doc_type": "contract"},
            files={"file": ("test.pdf", b"%PDF-1.7", "application/pdf")},
        )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Extraction failed" in response.json()["detail"]
