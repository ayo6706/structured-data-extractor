import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pymupdf
import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_db, get_document_service
from app.core.config import cost_settings
from app.main import app
from app.models.extraction import ExtractionStatus
from app.schemas.requests import ExtractionStrategy
from app.schemas.responses import AsyncExtractionResponse, ExtractResponse
from app.services.document_processor import DocumentProcessingResult
from app.services.documents import BATCH_CONCURRENCY_LIMIT, DocumentService
from app.services.extraction_pipeline import PipelineResult
from app.services.types import StrategyExtractionResult
from app.services.validation import ValidationResult


def _create_test_pdf(pages_text: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def _extract_response(
    *,
    document_id=None,
    extraction_id=None,
    warnings=None,
) -> ExtractResponse:
    return ExtractResponse(
        extraction_id=extraction_id or uuid4(),
        document_id=document_id or uuid4(),
        doc_type="invoice",
        status="completed",
        extracted_data={"vendor_name": "Acme"},
        confidence_map={"vendor_name": 0.9},
        warnings=warnings or [],
        model_used="test-model",
        input_tokens=10,
        output_tokens=5,
    )


def _processor_factory(processor: AsyncMock) -> MagicMock:
    processor.storage = getattr(processor, "storage", AsyncMock())
    processor.storage.save.return_value = "path/to/file"
    processor.storage.load.return_value = _create_test_pdf(["Page 1 content"])
    factory = MagicMock()
    factory.storage = processor.storage
    factory.create.return_value = processor
    audit = AsyncMock()
    audit.record_success.return_value = _extract_response()
    factory.create_audit_recorder.return_value = audit
    return factory


def _processing_result() -> DocumentProcessingResult:
    return DocumentProcessingResult(
        pipeline=PipelineResult(
            doc_type="invoice",
            validation=ValidationResult(
                extracted_data={"vendor_name": "Acme"},
                warnings=[],
                status=ExtractionStatus.COMPLETED,
            ),
            extraction=StrategyExtractionResult(
                raw_output={"vendor_name": "Acme"},
                input_tokens=10,
                output_tokens=5,
                strategy=ExtractionStrategy.FULL,
                source_pages=[0],
            ),
            confidence_map={"vendor_name": 0.9},
            warnings=[],
            duration_ms=10,
        ),
        classifier_usage_events=[],
    )


def _batch_service(
    processor: AsyncMock,
    session_factory: "_SessionFactory",
    arq_pool: AsyncMock | None = None,
) -> DocumentService:
    processor.process_pages.return_value = _processing_result()
    processor_factory = _processor_factory(processor)

    return DocumentService(
        db=AsyncMock(),
        processor=processor,
        processor_factory=processor_factory,
        arq_pool=arq_pool,
        session_factory=session_factory,
        price_for_model=cost_settings.price_for_model,
    )


class _SessionContext:
    def __init__(self, db: AsyncMock) -> None:
        self.db = db

    async def __aenter__(self) -> AsyncMock:
        return self.db

    async def __aexit__(self, _exc_type, _exc, _traceback) -> bool:
        return False


class _SessionFactory:
    def __init__(self) -> None:
        self.sessions: list[AsyncMock] = []

    def __call__(self) -> _SessionContext:
        db = AsyncMock()
        db.add = MagicMock()
        self.sessions.append(db)
        return _SessionContext(db)


@pytest.fixture
def mock_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    AsyncMock,
    AsyncMock,
    AsyncMock,
    _SessionFactory,
    DocumentService,
]:
    db = AsyncMock()
    db.add = MagicMock()
    extraction_service = AsyncMock()
    extraction_service.process_pages.return_value = _processing_result()
    processor_factory = _processor_factory(extraction_service)
    arq_pool = AsyncMock()
    session_factory = _SessionFactory()
    document_service = DocumentService(
        db=db,
        processor=extraction_service,
        processor_factory=processor_factory,
        arq_pool=arq_pool,
        session_factory=session_factory,
        price_for_model=cost_settings.price_for_model,
    )
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_document_service] = lambda: document_service
    monkeypatch.setattr(
        "app.api.dependencies.get_session_factory",
        lambda: session_factory,
    )

    try:
        yield (
            db,
            extraction_service,
            arq_pool,
            session_factory,
            document_service,
        )
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_document_service, None)


@pytest.mark.asyncio
async def test_batch_extract_async_success(
    mock_deps: tuple[
        AsyncMock,
        AsyncMock,
        AsyncMock,
        _SessionFactory,
        DocumentService,
    ],
) -> None:
    _, extraction_service, arq_pool, _, _ = mock_deps

    job_mock = SimpleNamespace(job_id="test-job-id")
    arq_pool.enqueue_job.return_value = job_mock

    extraction_service.storage.save.return_value = "path/to/file"

    pdf_bytes = _create_test_pdf(["Page 1 content"])

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        files = [
            ("files", ("test1.pdf", pdf_bytes, "application/pdf")),
            ("files", ("test2.pdf", pdf_bytes, "application/pdf")),
            ("files", ("test3.pdf", pdf_bytes, "application/pdf")),
        ]
        response = await client.post(
            "/api/v1/documents/extractions/batch",
            files=files,
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total"] == 3
    assert data["succeeded"] == 3
    assert data["failed"] == 0
    assert len(data["results"]) == 3
    for item in data["results"]:
        assert item["status"] == "processing"
        assert item["job_id"] == "test-job-id"
        assert item["document_id"] is not None


@pytest.mark.asyncio
async def test_batch_extract_sync_success(
    mock_deps: tuple[
        AsyncMock,
        AsyncMock,
        AsyncMock,
        _SessionFactory,
        DocumentService,
    ],
) -> None:
    (
        request_db,
        extraction_service,
        _,
        session_factory,
        document_service,
    ) = mock_deps

    extraction_service.process_pages.return_value = _processing_result()

    pdf_bytes = _create_test_pdf(["Page 1 content"])

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        files = [
            ("files", ("test1.pdf", pdf_bytes, "application/pdf")),
            ("files", ("test2.pdf", pdf_bytes, "application/pdf")),
        ]
        response = await client.post(
            "/api/v1/documents/extractions/batch",
            params={"async": "false"},
            files=files,
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total"] == 2
    assert data["succeeded"] == 2
    assert data["failed"] == 0
    assert len(data["results"]) == 2
    assert data["results"][0]["status"] == "completed"
    assert data["results"][0]["extraction_id"] is not None
    assert data["results"][0]["document_id"] is not None
    assert data["results"][0]["extracted_data"] == {"vendor_name": "Acme"}
    assert extraction_service.process_pages.await_count == 2
    assert len(session_factory.sessions) == 2
    assert all(
        used_db is not request_db for used_db in session_factory.sessions
    )


@pytest.mark.asyncio
async def test_batch_extract_size_limit(
    mock_deps: tuple[
        AsyncMock,
        AsyncMock,
        AsyncMock,
        _SessionFactory,
        DocumentService,
    ],
) -> None:
    pdf_bytes = _create_test_pdf(["Page 1 content"])
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        files = [
            ("files", (f"test{i}.pdf", pdf_bytes, "application/pdf"))
            for i in range(21)
        ]
        response = await client.post(
            "/api/v1/documents/extractions/batch",
            files=files,
        )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Maximum batch size is 20 files" in response.json()["detail"]


@pytest.mark.asyncio
async def test_batch_extract_rejects_non_pdf(
    mock_deps: tuple[
        AsyncMock,
        AsyncMock,
        AsyncMock,
        _SessionFactory,
        DocumentService,
    ],
) -> None:
    pdf_bytes = _create_test_pdf(["Page 1 content"])
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        files = [
            ("files", ("test1.pdf", pdf_bytes, "application/pdf")),
            ("files", ("test2.txt", b"hello", "text/plain")),
        ]
        response = await client.post(
            "/api/v1/documents/extractions/batch",
            files=files,
        )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Only PDF uploads are supported" in response.json()["detail"]


@pytest.mark.asyncio
async def test_batch_extract_partial_failure_sync(
    mock_deps: tuple[
        AsyncMock,
        AsyncMock,
        AsyncMock,
        _SessionFactory,
        DocumentService,
    ],
) -> None:
    _, extraction_service, _, _, _ = mock_deps

    async def process_side_effect(pages, page_count, doc_type, strategy):
        if "fail" in pages[0]:
            raise Exception("Failure mock")
        return _processing_result()

    extraction_service.process_pages.side_effect = process_side_effect

    pdf_bytes = _create_test_pdf(["Page 1 content"])
    fail_pdf_bytes = _create_test_pdf(["fail"])

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        files = [
            ("files", ("ok.pdf", pdf_bytes, "application/pdf")),
            ("files", ("fail.pdf", fail_pdf_bytes, "application/pdf")),
        ]
        response = await client.post(
            "/api/v1/documents/extractions/batch",
            params={"async": "false"},
            files=files,
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total"] == 2
    assert data["succeeded"] == 1
    assert data["failed"] == 1

    r0 = data["results"][0]
    r1 = data["results"][1]
    assert r0["filename"] == "ok.pdf"
    assert r0["status"] == "completed"

    assert r1["filename"] == "fail.pdf"
    assert r1["status"] == "failed"
    assert r1["document_id"] is None
    assert r1["error"] == "Failure mock"


@pytest.mark.asyncio
async def test_batch_extract_redis_down_fallback(
    mock_deps: tuple[
        AsyncMock,
        AsyncMock,
        AsyncMock,
        _SessionFactory,
        DocumentService,
    ],
) -> None:
    _, extraction_service, arq_pool, _, document_service = mock_deps
    document_service.arq_pool = None

    extraction_service.process_pages.return_value = _processing_result()

    pdf_bytes = _create_test_pdf(["Page 1 content"])

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            files = [
                ("files", ("test.pdf", pdf_bytes, "application/pdf")),
            ]
            response = await client.post(
                "/api/v1/documents/extractions/batch",
                params={"async": "true"},
                files=files,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 1
        assert data["succeeded"] == 1
        assert data["failed"] == 0
        assert data["results"][0]["status"] == "completed"
        warnings = data["results"][0]["warnings"]
        assert "Redis unavailable, processed inline" in warnings
    finally:
        document_service.arq_pool = arq_pool


@pytest.mark.asyncio
async def test_extract_async_enqueue_failure_reuses_created_document(
    mock_deps: tuple[
        AsyncMock,
        AsyncMock,
        AsyncMock,
        _SessionFactory,
        DocumentService,
    ],
) -> None:
    _, extraction_service, arq_pool, _, _ = mock_deps
    extraction_service.storage.save.return_value = "path/to/file"
    arq_pool.enqueue_job.side_effect = RuntimeError("redis down")

    extraction_service.process_pages.return_value = _processing_result()

    pdf_bytes = _create_test_pdf(["Page 1 content"])

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/documents/extractions",
            params={"async": "true"},
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "completed"
    assert "Redis unavailable, processed inline" in data["warnings"]
    extraction_service.process_pages.assert_awaited_once()


@pytest.mark.asyncio
async def test_extract_async_rejects_invalid_doc_type_before_enqueue(
    mock_deps: tuple[
        AsyncMock,
        AsyncMock,
        AsyncMock,
        _SessionFactory,
        DocumentService,
    ],
) -> None:
    _, extraction_service, arq_pool, _, _ = mock_deps
    pdf_bytes = _create_test_pdf(["Page 1 content"])

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/documents/extractions",
            params={"async": "true", "doc_type": "banana"},
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid doc_type 'banana'" in response.json()["detail"]
    extraction_service.storage.save.assert_not_called()
    arq_pool.enqueue_job.assert_not_called()


@pytest.mark.asyncio
async def test_extract_async_deletes_upload_when_document_commit_fails(
    mock_deps: tuple[
        AsyncMock,
        AsyncMock,
        AsyncMock,
        _SessionFactory,
        DocumentService,
    ],
) -> None:
    db, extraction_service, arq_pool, _, _ = mock_deps
    extraction_service.storage.save.return_value = "path/to/file"
    db.commit.side_effect = RuntimeError("database unavailable")

    pdf_bytes = _create_test_pdf(["Page 1 content"])

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/documents/extractions",
            params={"async": "true", "doc_type": "invoice"},
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    extraction_service.storage.delete.assert_awaited_once_with("path/to/file")
    arq_pool.enqueue_job.assert_not_called()
    extraction_service.process_pages.assert_not_called()


@pytest.mark.asyncio
async def test_batch_async_enqueue_failure_falls_back_per_item(
    mock_deps: tuple[
        AsyncMock,
        AsyncMock,
        AsyncMock,
        _SessionFactory,
        DocumentService,
    ],
) -> None:
    _, extraction_service, arq_pool, _, _ = mock_deps
    extraction_service.storage.save.return_value = "path/to/file"

    job_mock = SimpleNamespace(job_id="queued-job")
    arq_pool.enqueue_job.side_effect = [job_mock, RuntimeError("redis down")]

    extraction_service.process_pages.return_value = _processing_result()

    pdf_bytes = _create_test_pdf(["Page 1 content"])

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        files = [
            ("files", ("queued.pdf", pdf_bytes, "application/pdf")),
            ("files", ("fallback.pdf", pdf_bytes, "application/pdf")),
        ]
        response = await client.post(
            "/api/v1/documents/extractions/batch",
            files=files,
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["succeeded"] == 2
    assert data["failed"] == 0
    assert data["results"][0]["status"] == "processing"
    assert data["results"][0]["job_id"] == "queued-job"
    assert data["results"][1]["status"] == "completed"
    assert (
        "Redis unavailable, processed inline" in data["results"][1]["warnings"]
    )


@pytest.mark.asyncio
async def test_batch_inline_failure_before_document_has_no_document_id():
    extraction_service = AsyncMock()
    session_factory = _SessionFactory()
    extraction_service.process_pages.side_effect = RuntimeError(
        "parse failed before document"
    )

    service = _batch_service(extraction_service, session_factory)
    response = await service.batch_extract(
        file_data=[("fail.pdf", b"%PDF-1.7")],
        doc_type="invoice",
        strategy=ExtractionStrategy.FULL,
        async_mode=False,
    )
    results = response.results

    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].document_id is None
    assert "Failed to parse PDF" in results[0].error


@pytest.mark.asyncio
async def test_batch_enqueue_failure_before_document_has_no_document_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_service = AsyncMock()
    session_factory = _SessionFactory()
    arq_pool = AsyncMock()

    async def extract_document(self, **kwargs):
        raise RuntimeError("document commit failed")

    monkeypatch.setattr(
        "app.services.documents.DocumentService.extract_document",
        extract_document,
    )

    service = _batch_service(extraction_service, session_factory, arq_pool)
    response = await service.batch_extract(
        file_data=[("fail.pdf", b"%PDF-1.7")],
        doc_type="invoice",
        strategy=ExtractionStrategy.FULL,
        async_mode=True,
    )
    results = response.results

    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].document_id is None
    assert results[0].error == "document commit failed"


@pytest.mark.asyncio
async def test_process_batch_inline_limits_concurrency() -> None:
    extraction_service = AsyncMock()
    session_factory = _SessionFactory()
    active = 0
    max_active = 0
    started = 0
    release = asyncio.Event()
    limit = BATCH_CONCURRENCY_LIMIT

    async def extract_side_effect(pages, page_count, doc_type, strategy):
        nonlocal active, max_active, started
        active += 1
        started += 1
        max_active = max(max_active, active)
        if started == limit:
            release.set()
        await release.wait()
        await asyncio.sleep(0)
        active -= 1
        return _processing_result()

    extraction_service.process_pages.side_effect = extract_side_effect

    service = _batch_service(extraction_service, session_factory)
    response = await service.batch_extract(
        file_data=[
            (f"doc-{index}.pdf", _create_test_pdf([f"doc {index}"]))
            for index in range(limit + 2)
        ],
        doc_type="invoice",
        strategy=ExtractionStrategy.FULL,
        async_mode=False,
    )
    results = response.results

    assert len(results) == limit + 2
    assert all(result.status == "completed" for result in results)
    assert max_active == limit
    assert len(session_factory.sessions) == limit + 2


@pytest.mark.asyncio
async def test_process_batch_enqueue_limits_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_service = AsyncMock()
    session_factory = _SessionFactory()
    active = 0
    max_active = 0
    started = 0
    release = asyncio.Event()
    limit = BATCH_CONCURRENCY_LIMIT

    async def extract_document(self, **kwargs):
        nonlocal active, max_active, started
        active += 1
        started += 1
        max_active = max(max_active, active)
        if started == limit:
            release.set()
        await release.wait()
        await asyncio.sleep(0)
        active -= 1
        return AsyncExtractionResponse(
            document_id=uuid4(),
            job_id="job-id",
            status="processing",
        )

    monkeypatch.setattr(
        "app.services.documents.DocumentService.extract_document",
        extract_document,
    )

    service = _batch_service(extraction_service, session_factory)
    response = await service.batch_extract(
        file_data=[
            (f"doc-{index}.pdf", b"%PDF-1.7") for index in range(limit + 2)
        ],
        doc_type="invoice",
        strategy=ExtractionStrategy.FULL,
        async_mode=True,
    )
    results = response.results

    assert len(results) == limit + 2
    assert all(result.status == "processing" for result in results)
    assert max_active == limit
    assert len(session_factory.sessions) == limit + 2
