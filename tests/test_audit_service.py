from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.exceptions import ExtractionError
from app.models.document import Document, DocumentStatus
from app.models.extraction import ExtractionStatus
from app.models.llm_usage import LLMUsagePurpose
from app.repositories.extractions import ExtractionRepository
from app.repositories.llm_usage import LLMUsageRepository
from app.schemas.requests import ExtractionStrategy
from app.services.audit import AuditRecorder
from app.services.extraction_pipeline import PipelineResult
from app.services.types import (
    LLMUsageEvent,
    StrategyExtractionResult,
)
from app.services.validation import ValidationResult


def _pipeline_result() -> PipelineResult:
    return PipelineResult(
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
        confidence_map={"vendor_name": 0.85},
        warnings=[],
        duration_ms=123,
    )


def _audit_recorder(db: AsyncMock) -> AuditRecorder:
    return AuditRecorder(
        db=db,
        extractions=ExtractionRepository(db),
        llm_usage=LLMUsageRepository(db),
        model="test-model",
    )


@pytest.mark.asyncio
async def test_audit_service_records_success(monkeypatch) -> None:
    db = AsyncMock()
    db.add = MagicMock()
    document = Document(
        id=uuid4(),
        filename="invoice.pdf",
        file_path="uploads/invoice.pdf",
        file_size_bytes=10,
        page_count=1,
        status=DocumentStatus.PROCESSING,
    )
    extraction = MagicMock()
    extraction.id = uuid4()

    async def create_extraction(*_args, **kwargs):
        extraction.kwargs = kwargs
        return extraction

    monkeypatch.setattr(
        ExtractionRepository,
        "create_from_fields",
        create_extraction,
    )
    usage_events = []

    async def create_usage(*_args, **kwargs):
        usage_events.append(kwargs)

    monkeypatch.setattr(LLMUsageRepository, "create", create_usage)

    response = await _audit_recorder(db).record_success(
        document=document,
        result=_pipeline_result(),
        classifier_usage_events=[
            LLMUsageEvent(
                purpose=LLMUsagePurpose.CLASSIFIER,
                model="classifier-model",
                input_tokens=3,
                output_tokens=1,
            )
        ],
    )

    assert document.status == DocumentStatus.PROCESSING
    assert extraction.kwargs["document_id"] == document.id
    assert extraction.kwargs["model_used"] == "test-model"
    assert response.extraction_id == extraction.id
    assert response.document_id == document.id
    assert response.status == "completed"
    assert [event["purpose"] for event in usage_events] == [
        LLMUsagePurpose.CLASSIFIER,
        LLMUsagePurpose.EXTRACTION,
    ]
    db.refresh.assert_awaited_once_with(extraction)


@pytest.mark.asyncio
async def test_audit_service_records_failure() -> None:
    db = AsyncMock()
    db.add = MagicMock()
    document = Document(
        id=uuid4(),
        filename="invoice.pdf",
        file_path="uploads/invoice.pdf",
        file_size_bytes=10,
        page_count=1,
        status=DocumentStatus.PROCESSING,
    )

    await _audit_recorder(db).record_failure(
        document=document,
        doc_type="invoice",
        strategy="full",
        started_at=0,
        exc=ExtractionError(
            "failed",
            attempts=1,
            input_tokens=1,
            output_tokens=2,
        ),
        classifier_usage_events=[],
    )

    assert document.status == DocumentStatus.PROCESSING
    assert db.add.call_count >= 2


@pytest.mark.asyncio
async def test_audit_service_does_not_commit_transactions() -> None:
    db = AsyncMock()
    db.add = MagicMock()
    document = Document(
        id=uuid4(),
        filename="invoice.pdf",
        file_path="uploads/invoice.pdf",
        file_size_bytes=10,
        page_count=1,
        status=DocumentStatus.PROCESSING,
    )

    await _audit_recorder(db).record_failure(
        document=document,
        doc_type="invoice",
        strategy="full",
        started_at=0,
        exc=ExtractionError(
            "failed",
            attempts=1,
            input_tokens=1,
            output_tokens=2,
        ),
        classifier_usage_events=[],
    )

    db.commit.assert_not_awaited()
    db.rollback.assert_not_awaited()
