from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.config import ModelTokenPrice
from app.core.exceptions import (
    DocumentNotFoundError,
    ExtractionNotFoundError,
    InvalidUploadError,
)
from app.models.document import DocumentStatus
from app.models.extraction import Extraction, ExtractionStatus
from app.services.documents import DocumentService


def _service(db: AsyncMock) -> DocumentService:
    processor = AsyncMock()
    processor.storage = AsyncMock()
    return DocumentService(
        db=db,
        processor=processor,
        storage=processor.storage,
        model="test-model",
        arq_pool=None,
        session_factory=AsyncMock(),
        price_for_model=lambda _model: ModelTokenPrice(
            input_token_price_usd=Decimal("0"),
            output_token_price_usd=Decimal("0"),
        ),
    )


def _mock_scalar(db: AsyncMock, value: object) -> None:
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    db.execute.return_value = result


@pytest.mark.asyncio
async def test_get_document_returns_response() -> None:
    db = AsyncMock()
    doc_id = uuid4()
    ext1_id = uuid4()
    ext2_id = uuid4()

    document = MagicMock()
    document.id = doc_id
    document.filename = "contract.pdf"
    document.status = DocumentStatus.COMPLETED
    document.page_count = 5
    document.file_size_bytes = 1048576
    document.uploaded_at = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
    document.extractions = [
        MagicMock(id=ext1_id),
        MagicMock(id=ext2_id),
    ]
    _mock_scalar(db, document)

    response = await _service(db).get_document(doc_id)

    assert response.id == doc_id
    assert response.filename == "contract.pdf"
    assert response.status == DocumentStatus.COMPLETED
    assert response.extraction_ids == [ext1_id, ext2_id]


@pytest.mark.asyncio
async def test_get_document_raises_for_missing_document() -> None:
    db = AsyncMock()
    doc_id = uuid4()
    _mock_scalar(db, None)

    with pytest.raises(DocumentNotFoundError):
        await _service(db).get_document(doc_id)


@pytest.mark.asyncio
async def test_get_extraction_returns_public_response() -> None:
    db = AsyncMock()
    extraction = _extraction()
    _mock_scalar(db, extraction)

    response = await _service(db).get_extraction(extraction.id)

    assert response.extraction_id == extraction.id
    assert response.document_id == extraction.document_id
    assert response.status == "completed"
    assert response.confidence_map == {"vendor_name": 0.85}


@pytest.mark.asyncio
async def test_get_extraction_handles_string_status_and_bad_confidence(
) -> None:
    db = AsyncMock()
    extraction = _extraction()
    extraction.status = "completed"
    extraction.confidence_map = {"vendor_name": "high", "currency": 0.85}
    _mock_scalar(db, extraction)

    response = await _service(db).get_extraction(extraction.id)

    assert response.status == "completed"
    assert response.confidence_map == {
        "vendor_name": 0.0,
        "currency": 0.85,
    }


@pytest.mark.asyncio
async def test_get_extraction_audit_returns_raw_fields() -> None:
    db = AsyncMock()
    extraction = _extraction()
    _mock_scalar(db, extraction)

    response = await _service(db).get_extraction_audit(extraction.id)

    assert response.raw_tool_output == {"vendor_name": "Acme Corp"}
    assert response.strategy == "full"
    assert response.source_pages == [0]


@pytest.mark.asyncio
async def test_get_extraction_raises_for_missing_extraction() -> None:
    db = AsyncMock()
    extraction_id = uuid4()
    _mock_scalar(db, None)

    with pytest.raises(ExtractionNotFoundError):
        await _service(db).get_extraction(extraction_id)


@pytest.mark.asyncio
async def test_extract_document_requires_exactly_one_source() -> None:
    service = _service(AsyncMock())

    with pytest.raises(InvalidUploadError):
        await service.extract_document(
            content=None,
            filename=None,
            document_id=None,
            doc_type=None,
            strategy=None,
            async_mode=False,
        )

    with pytest.raises(InvalidUploadError):
        await service.extract_document(
            content=b"%PDF-1.7",
            filename="upload.pdf",
            document_id=uuid4(),
            doc_type=None,
            strategy=None,
            async_mode=False,
        )


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
