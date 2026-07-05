from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.workers.worker import WorkerSettings, extraction_job


@pytest.mark.asyncio
async def test_extraction_job_success(monkeypatch: pytest.MonkeyPatch) -> None:
    db_mock = AsyncMock()

    session_factory_mock = MagicMock()
    session_factory_mock.return_value.__aenter__.return_value = db_mock
    session_factory_mock.return_value.__aexit__.return_value = False

    processor_mock = AsyncMock()
    tool_extractor = MagicMock(model="test-model")
    storage = AsyncMock()
    document_service = AsyncMock()
    document_service_cls = MagicMock(return_value=document_service)
    monkeypatch.setattr(
        "app.workers.worker.DocumentService",
        document_service_cls,
    )

    ctx = {
        "session_factory": session_factory_mock,
        "processor": processor_mock,
        "storage": storage,
        "tool_extractor": tool_extractor,
    }

    doc_id = uuid4()
    await extraction_job(
        ctx=ctx,
        document_id=str(doc_id),
        doc_type="invoice",
        strategy="smart",
    )

    document_service._process_existing_inline.assert_called_once_with(
        document_id=doc_id,
        doc_type="invoice",
        strategy="smart",
    )


@pytest.mark.asyncio
async def test_extraction_job_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    db_mock = AsyncMock()

    session_factory_mock = MagicMock()
    session_factory_mock.return_value.__aenter__.return_value = db_mock
    session_factory_mock.return_value.__aexit__.return_value = False

    processor_mock = AsyncMock()
    tool_extractor = MagicMock(model="test-model")
    storage = AsyncMock()
    document_service = AsyncMock()
    document_service_cls = MagicMock(return_value=document_service)
    monkeypatch.setattr(
        "app.workers.worker.DocumentService",
        document_service_cls,
    )
    ext_mock = document_service._process_existing_inline
    ext_mock.side_effect = Exception("Pipeline error")

    ctx = {
        "session_factory": session_factory_mock,
        "processor": processor_mock,
        "storage": storage,
        "tool_extractor": tool_extractor,
    }

    doc_id = uuid4()
    with pytest.raises(Exception, match="Pipeline error"):
        await extraction_job(
            ctx=ctx,
            document_id=str(doc_id),
            doc_type="payslip",
            strategy=None,
        )

    document_service._process_existing_inline.assert_called_once_with(
        document_id=doc_id,
        doc_type="payslip",
        strategy=None,
    )


def test_worker_limits_concurrent_jobs() -> None:
    assert WorkerSettings.max_jobs == 5
