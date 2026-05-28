from unittest.mock import AsyncMock, MagicMock

import fitz
import pytest

from app.core.exceptions import (
    ExtractionError,
    PDFParseError,
    UnsupportedDocumentTypeError,
)
from app.integrations.llm.client import LLMClientError, ToolCallResult
from app.models.extraction import Extraction, ExtractionStatus
from app.schemas.requests import ExtractionStrategy
from app.services.extraction import EXTRACTION_SYSTEM_PROMPT, ExtractionService


def _create_test_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def _create_zero_page_pdf() -> bytes:
    return (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [] /Count 0 >> endobj\n"
        b"xref\n"
        b"0 3\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"trailer << /Root 1 0 R /Size 3 >>\n"
        b"startxref\n"
        b"111\n"
        b"%%EOF\n"
    )


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def extraction_service(mock_llm_client: AsyncMock) -> ExtractionService:
    storage = AsyncMock()
    storage.save.return_value = "test-id/test.pdf"
    return ExtractionService(
        classifier=AsyncMock(),
        llm_client=mock_llm_client,
        storage=storage,
        model="test-model",
    )


@pytest.mark.asyncio
async def test_extract_from_text_success_first_try(
    extraction_service: ExtractionService,
    mock_llm_client: AsyncMock,
) -> None:
    expected_data = {"vendor_name": "Acme Corp", "total_amount": 100.0}
    mock_llm_client.call_tool.return_value = ToolCallResult(
        arguments=expected_data,
        input_tokens=100,
        output_tokens=50,
    )

    result, input_tokens, output_tokens = (
        await extraction_service.extract_from_text(
            text="Acme Corp invoice...", doc_type="invoice"
        )
    )

    assert result == expected_data
    assert input_tokens == 100
    assert output_tokens == 50
    mock_llm_client.call_tool.assert_called_once()


@pytest.mark.asyncio
async def test_extract_from_text_success_after_retry(
    extraction_service: ExtractionService,
    mock_llm_client: AsyncMock,
) -> None:
    expected_data = {"vendor_name": "Acme Corp"}
    mock_llm_client.call_tool.side_effect = [
        None,
        ToolCallResult(
            arguments=expected_data, input_tokens=150, output_tokens=50
        ),
    ]

    result, input_tokens, output_tokens = (
        await extraction_service.extract_from_text(
            text="Acme Corp invoice...", doc_type="invoice"
        )
    )

    assert result == expected_data
    assert input_tokens == 150
    assert output_tokens == 50
    assert mock_llm_client.call_tool.call_count == 2


@pytest.mark.asyncio
async def test_extract_from_text_exhausted_retries(
    mock_llm_client: AsyncMock,
) -> None:
    mock_llm_client.call_tool.return_value = None
    service = ExtractionService(
        classifier=AsyncMock(),
        llm_client=mock_llm_client,
        storage=AsyncMock(),
        max_retries=2,
    )

    with pytest.raises(ExtractionError) as exc_info:
        await service.extract_from_text(
            text="Acme Corp invoice...", doc_type="invoice"
        )

    assert exc_info.value.attempts == 3
    assert "after 3 attempts" in str(exc_info.value)
    assert mock_llm_client.call_tool.call_count == 3


@pytest.mark.asyncio
async def test_extract_from_text_passes_tool_inputs(
    extraction_service: ExtractionService,
    mock_llm_client: AsyncMock,
) -> None:
    mock_llm_client.call_tool.return_value = ToolCallResult(
        arguments={"vendor_name": "Acme Corp"},
        input_tokens=100,
        output_tokens=50,
    )

    await extraction_service.extract_from_text(
        text="Acme Corp invoice...", doc_type="invoice"
    )

    call = mock_llm_client.call_tool.call_args
    assert call.kwargs["model"] == "test-model"
    assert call.kwargs["tool_name"] == "extract_invoice"
    assert call.kwargs["tool"]["function"]["name"] == "extract_invoice"
    assert call.kwargs["system_prompt"] == EXTRACTION_SYSTEM_PROMPT
    assert call.kwargs["user_content"] == (
        "Document text:\nAcme Corp invoice..."
    )


@pytest.mark.asyncio
async def test_extract_from_text_wraps_llm_client_error(
    extraction_service: ExtractionService,
    mock_llm_client: AsyncMock,
) -> None:
    mock_llm_client.call_tool.side_effect = LLMClientError("bad response")

    with pytest.raises(ExtractionError) as exc_info:
        await extraction_service.extract_from_text(
            text="Acme Corp invoice...", doc_type="invoice"
        )

    assert exc_info.value.attempts == 1
    assert isinstance(exc_info.value.original_exception, LLMClientError)


@pytest.mark.asyncio
async def test_extract_from_pages_full_uses_all_pages(
    extraction_service: ExtractionService,
) -> None:
    extraction_service.extract_from_text = AsyncMock(
        return_value=(
            {"vendor_name": "Acme Corp"},
            100,
            50,
        )
    )

    result = await extraction_service.extract_from_pages(
        pages=["First page", "Second page"],
        doc_type="invoice",
        strategy=ExtractionStrategy.FULL,
    )

    assert result.strategy == ExtractionStrategy.FULL
    assert result.source_pages == [0, 1]
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    call = extraction_service.extract_from_text.call_args
    assert "Page 0:\nFirst page" in call.kwargs["text"]
    assert "Page 1:\nSecond page" in call.kwargs["text"]


def test_resolve_strategy_defaults_to_full_for_backwards_compatibility() -> None:
    assert (
        ExtractionService._resolve_strategy(strategy=None)
        == ExtractionStrategy.FULL
    )
    assert (
        ExtractionService._resolve_strategy(
            strategy="smart",
        )
        == ExtractionStrategy.SMART
    )


def test_select_pages_includes_first_last_and_monetary_pages() -> None:
    pages = [
        "Cover",
        "Terms only",
        "Total amount due is USD 50",
        "Signature",
    ]

    selected = ExtractionService._select_pages(pages)

    assert selected == [
        (0, "Cover"),
        (2, "Total amount due is USD 50"),
        (3, "Signature"),
    ]


@pytest.mark.asyncio
async def test_page_by_page_merges_schema_fields_and_tracks_sources(
    extraction_service: ExtractionService,
) -> None:
    extraction_service.extract_from_text = AsyncMock(
        side_effect=[
            (
                {
                    "parties": [{"name": "Client Corp", "role": "Client"}],
                    "effective_date": "2026-05-27",
                    "key_obligations": ["Deliver goods"],
                },
                10,
                5,
            ),
            (
                {
                    "parties": [{"name": "Vendor Corp", "role": "Vendor"}],
                    "payment_amount": "1500.00",
                    "key_obligations": ["Support rollout"],
                },
                12,
                6,
            ),
        ]
    )

    result = await extraction_service.extract_from_pages(
        pages=["Client terms", "Payment terms"],
        doc_type="contract",
        strategy=ExtractionStrategy.PAGE_BY_PAGE,
    )

    assert result.raw_output == {
        "parties": [
            {"name": "Client Corp", "role": "Client"},
            {"name": "Vendor Corp", "role": "Vendor"},
        ],
        "effective_date": "2026-05-27",
        "key_obligations": ["Deliver goods", "Support rollout"],
        "payment_amount": "1500.00",
    }
    assert result.source_pages == [0, 1]
    assert result.field_source_pages == {
        "parties": [0, 1],
        "effective_date": [0],
        "key_obligations": [0, 1],
        "payment_amount": [1],
    }
    assert result.input_tokens == 22
    assert result.output_tokens == 11


@pytest.mark.asyncio
async def test_page_by_page_replaces_scalar_source_page_on_better_value(
    extraction_service: ExtractionService,
) -> None:
    extraction_service.extract_from_text = AsyncMock(
        side_effect=[
            ({"governing_law": "NY"}, 10, 5),
            ({"governing_law": "New York State"}, 12, 6),
        ]
    )

    result = await extraction_service.extract_from_pages(
        pages=["Short law", "Specific law"],
        doc_type="contract",
        strategy=ExtractionStrategy.PAGE_BY_PAGE,
    )

    assert result.raw_output == {"governing_law": "New York State"}
    assert result.source_pages == [1]
    assert result.field_source_pages == {"governing_law": [1]}


@pytest.mark.asyncio
async def test_page_by_page_skips_failed_pages_when_others_succeed(
    extraction_service: ExtractionService,
) -> None:
    extraction_service.extract_from_text = AsyncMock(
        side_effect=[
            ExtractionError(
                "page failed",
                attempts=1,
                input_tokens=7,
                output_tokens=3,
            ),
            (
                {
                    "parties": [{"name": "Client Corp"}],
                    "effective_date": "2026-05-27",
                    "key_obligations": ["Deliver goods"],
                },
                10,
                5,
            ),
        ]
    )

    result = await extraction_service.extract_from_pages(
        pages=["Bad page", "Good page"],
        doc_type="contract",
        strategy=ExtractionStrategy.PAGE_BY_PAGE,
    )

    assert result.source_pages == [1]
    assert result.input_tokens == 17
    assert result.output_tokens == 8
    assert result.warnings == [
        "page 0: extraction failed and was skipped"
    ]


@pytest.mark.asyncio
async def test_extract_upload_persists_failed_extraction(
    mock_llm_client: AsyncMock,
) -> None:
    db = AsyncMock()
    db.add = MagicMock()
    storage = AsyncMock()
    storage.save.return_value = "test-id/test.pdf"
    service = ExtractionService(
        classifier=AsyncMock(),
        llm_client=mock_llm_client,
        storage=storage,
        max_retries=0,
    )
    mock_llm_client.call_tool.return_value = None

    with pytest.raises(ExtractionError):
        await service.extract_upload(
            content=_create_test_pdf("Some contract data"),
            filename="test.pdf",
            doc_type="contract",
            db=db,
        )

    failed_extractions = [
        call.args[0]
        for call in db.add.call_args_list
        if isinstance(call.args[0], Extraction)
        and call.args[0].status == ExtractionStatus.FAILED
    ]
    assert len(failed_extractions) == 1
    assert db.commit.call_count >= 3


@pytest.mark.asyncio
async def test_extract_upload_records_tokens_from_extraction_error(
    mock_llm_client: AsyncMock,
) -> None:
    db = AsyncMock()
    db.add = MagicMock()
    storage = AsyncMock()
    storage.save.return_value = "test-id/test.pdf"
    service = ExtractionService(
        classifier=AsyncMock(),
        llm_client=mock_llm_client,
        storage=storage,
    )
    service.extract_from_text = AsyncMock(
        side_effect=ExtractionError(
            "LLM failed",
            attempts=2,
            input_tokens=123,
            output_tokens=45,
        )
    )

    with pytest.raises(ExtractionError):
        await service.extract_upload(
            content=_create_test_pdf("Some contract data"),
            filename="test.pdf",
            doc_type="contract",
            db=db,
        )

    failed_extraction = next(
        call.args[0]
        for call in db.add.call_args_list
        if isinstance(call.args[0], Extraction)
        and call.args[0].status == ExtractionStatus.FAILED
    )
    assert failed_extraction.input_tokens == 123
    assert failed_extraction.output_tokens == 45


@pytest.mark.asyncio
async def test_extract_upload_persists_failed_validation_result(
    mock_llm_client: AsyncMock,
) -> None:
    db = AsyncMock()
    db.add = MagicMock()
    storage = AsyncMock()
    storage.save.return_value = "test-id/test.pdf"
    service = ExtractionService(
        classifier=AsyncMock(),
        llm_client=mock_llm_client,
        storage=storage,
    )
    mock_llm_client.call_tool.return_value = ToolCallResult(
        arguments={"invalid_field": "some data"},
        input_tokens=50,
        output_tokens=10,
    )

    response = await service.extract_upload(
        content=_create_test_pdf("Some contract data"),
        filename="test.pdf",
        doc_type="contract",
        db=db,
    )

    failed_extractions = [
        call.args[0]
        for call in db.add.call_args_list
        if isinstance(call.args[0], Extraction)
        and call.args[0].status == ExtractionStatus.FAILED
    ]
    assert len(failed_extractions) == 1
    assert failed_extractions[0].raw_tool_output == {
        "invalid_field": "some data"
    }
    assert failed_extractions[0].warnings
    assert response.status == "failed"
    assert response.extracted_data is None


@pytest.mark.asyncio
async def test_extract_upload_returns_result_when_audit_persist_fails(
    mock_llm_client: AsyncMock,
) -> None:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush.side_effect = RuntimeError("database down")
    storage = AsyncMock()
    storage.save.return_value = "test-id/test.pdf"
    service = ExtractionService(
        classifier=AsyncMock(),
        llm_client=mock_llm_client,
        storage=storage,
    )
    mock_llm_client.call_tool.return_value = ToolCallResult(
        arguments={
            "parties": [
                {"name": "Client Corp", "role": "Client"},
                {"name": "Vendor Corp", "role": "Vendor"},
            ],
            "effective_date": "2026-05-27",
            "key_obligations": ["Deliver goods"],
        },
        input_tokens=100,
        output_tokens=50,
    )

    response = await service.extract_upload(
        content=_create_test_pdf("Some contract data"),
        filename="test.pdf",
        doc_type="contract",
        db=db,
    )

    assert response.extraction_id is None
    assert response.status == "completed"
    assert response.extracted_data["effective_date"] == "2026-05-27"
    assert response.confidence_map["parties"] == 0.85
    assert any(
        isinstance(call.args[0], Extraction) for call in db.add.call_args_list
    )
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_extract_upload_validates_explicit_doc_type_before_side_effects(
    mock_llm_client: AsyncMock,
) -> None:
    db = AsyncMock()
    db.add = MagicMock()
    storage = AsyncMock()
    service = ExtractionService(
        classifier=AsyncMock(),
        llm_client=mock_llm_client,
        storage=storage,
    )

    with pytest.raises(UnsupportedDocumentTypeError) as exc_info:
        await service.extract_upload(
            content=_create_test_pdf("Some data"),
            filename="test.pdf",
            doc_type="banana",
            db=db,
        )

    assert "Unsupported document type: banana" in str(exc_info.value)
    storage.save.assert_not_called()
    db.add.assert_not_called()
    db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_extract_upload_rejects_zero_page_pdf_before_side_effects(
    mock_llm_client: AsyncMock,
) -> None:
    db = AsyncMock()
    db.add = MagicMock()
    storage = AsyncMock()
    service = ExtractionService(
        classifier=AsyncMock(),
        llm_client=mock_llm_client,
        storage=storage,
    )

    with pytest.raises(PDFParseError) as exc_info:
        await service.extract_upload(
            content=_create_zero_page_pdf(),
            filename="empty.pdf",
            doc_type="contract",
            db=db,
        )

    assert "PDF does not contain any pages" in str(exc_info.value)
    storage.save.assert_not_called()
    db.add.assert_not_called()
    db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_extract_upload_offloads_pdf_parsing(
    mock_llm_client: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = AsyncMock()
    db.add = MagicMock()
    storage = AsyncMock()
    storage.save.return_value = "test-id/test.pdf"
    service = ExtractionService(
        classifier=AsyncMock(),
        llm_client=mock_llm_client,
        storage=storage,
    )
    mock_llm_client.call_tool.return_value = ToolCallResult(
        arguments={
            "parties": [
                {"name": "Client Corp", "role": "Client"},
                {"name": "Vendor Corp", "role": "Vendor"},
            ],
            "effective_date": "2026-05-27",
            "key_obligations": ["Deliver goods"],
        },
        input_tokens=100,
        output_tokens=50,
    )

    run_sync = AsyncMock()

    async def fake_run_sync(func, *args):
        return func(*args)

    run_sync.side_effect = fake_run_sync
    monkeypatch.setattr(
        "app.services.extraction.anyio.to_thread.run_sync", run_sync
    )

    await service.extract_upload(
        content=_create_test_pdf("Some contract data"),
        filename="test.pdf",
        doc_type="contract",
        db=db,
    )

    run_sync.assert_called_once()


@pytest.mark.asyncio
async def test_extract_upload_deletes_saved_file_if_initial_db_commit_fails(
    mock_llm_client: AsyncMock,
) -> None:
    db = AsyncMock()
    db.add = MagicMock()
    db.commit.side_effect = RuntimeError("database down")
    storage = AsyncMock()
    storage.save.return_value = "test-id/test.pdf"
    service = ExtractionService(
        classifier=AsyncMock(),
        llm_client=mock_llm_client,
        storage=storage,
    )

    with pytest.raises(RuntimeError, match="database down"):
        await service.extract_upload(
            content=_create_test_pdf("Some contract data"),
            filename="test.pdf",
            doc_type="contract",
            db=db,
        )

    storage.delete.assert_called_once_with("test-id/test.pdf")
