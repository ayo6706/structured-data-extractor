from types import SimpleNamespace
from unittest.mock import AsyncMock

import fitz
import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_classifier_service
from app.core.exceptions import ClassificationError
from app.main import app


def _create_test_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.fixture
def mock_classifier() -> AsyncMock:
    classifier = AsyncMock()
    previous_override = app.dependency_overrides.get(get_classifier_service)
    app.dependency_overrides[get_classifier_service] = lambda: classifier
    try:
        yield classifier
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(get_classifier_service, None)
        else:
            app.dependency_overrides[get_classifier_service] = (
                previous_override
            )


@pytest.mark.asyncio
async def test_extract_explicit_doc_type_success(
    mock_classifier: AsyncMock,
) -> None:
    pdf_bytes = _create_test_pdf("Some contract data")
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            params={"doc_type": "contract"},
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"doc_type": "contract"}
    mock_classifier.classify.assert_not_called()


@pytest.mark.asyncio
async def test_extract_explicit_doc_type_invalid(
    mock_classifier: AsyncMock,
) -> None:
    pdf_bytes = _create_test_pdf("Some invalid doc_type data")
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            params={"doc_type": "banana"},
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid doc_type 'banana'" in response.json()["detail"]
    mock_classifier.classify.assert_not_called()


@pytest.mark.asyncio
async def test_extract_auto_success(mock_classifier: AsyncMock) -> None:
    mock_classifier.classify.return_value = "receipt"
    pdf_bytes = _create_test_pdf("Gas station receipt content")
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"doc_type": "receipt"}
    mock_classifier.classify.assert_called_once()
    assert (
        "Gas station receipt content"
        in mock_classifier.classify.call_args[0][0]
    )


@pytest.mark.asyncio
async def test_extract_auto_unknown(mock_classifier: AsyncMock) -> None:
    mock_classifier.classify.return_value = "unknown"
    pdf_bytes = _create_test_pdf("Random document with unknown type")
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert "Could not determine the document type" in response.json()["detail"]
    mock_classifier.classify.assert_called_once()


@pytest.mark.asyncio
async def test_extract_classification_error(
    mock_classifier: AsyncMock,
) -> None:
    mock_classifier.classify.side_effect = ClassificationError("LLM failed")
    pdf_bytes = _create_test_pdf("Failed document text")
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert "Classification service unavailable" in response.json()["detail"]
    mock_classifier.classify.assert_called_once()


@pytest.mark.asyncio
async def test_extract_invalid_pdf(mock_classifier: AsyncMock) -> None:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("test.pdf", b"garbage bytes", "application/pdf")},
        )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Could not parse PDF"
    mock_classifier.classify.assert_not_called()


@pytest.mark.asyncio
async def test_extract_rejects_non_pdf_upload(
    mock_classifier: AsyncMock,
) -> None:
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
    mock_classifier.classify.assert_not_called()


@pytest.mark.asyncio
async def test_extract_rejects_oversized_upload(
    mock_classifier: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.api.v1.endpoints.extract.get_app_settings",
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
    mock_classifier.classify.assert_not_called()
