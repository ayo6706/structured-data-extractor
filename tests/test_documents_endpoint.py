from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_db, get_document_service
from app.core.exceptions import DocumentNotFoundError
from app.main import app
from app.schemas.responses import DocumentResponse


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
async def test_get_document_success(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    mock_db, document_service = mock_deps
    doc_id = uuid4()
    ext1_id = uuid4()
    ext2_id = uuid4()

    document_service.get_document.return_value = DocumentResponse(
        id=doc_id,
        filename="contract.pdf",
        status="completed",
        page_count=5,
        file_size_bytes=1048576,
        uploaded_at=datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC),
        extraction_ids=[ext1_id, ext2_id],
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/documents/{doc_id}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["id"] == str(doc_id)
    assert data["filename"] == "contract.pdf"
    assert data["status"] == "completed"
    assert data["page_count"] == 5
    assert data["file_size_bytes"] == 1048576
    assert data["uploaded_at"] == "2026-05-28T12:00:00Z"
    assert data["extraction_ids"] == [str(ext1_id), str(ext2_id)]

    document_service.get_document.assert_called_once_with(doc_id)


@pytest.mark.asyncio
async def test_get_document_not_found(
    mock_deps: tuple[AsyncMock, AsyncMock],
) -> None:
    mock_db, document_service = mock_deps
    doc_id = uuid4()
    document_service.get_document.side_effect = DocumentNotFoundError(doc_id)

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/documents/{doc_id}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "Document not found"
    document_service.get_document.assert_called_once_with(doc_id)
