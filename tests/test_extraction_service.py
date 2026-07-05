from unittest.mock import AsyncMock, MagicMock

import fitz
import pytest

from app.core.exceptions import (
    ExtractionError,
    PDFParseError,
    UnsupportedDocumentTypeError,
)
from app.integrations.llm.client import LLMClientError, ToolCallResult
from app.models.document import DocumentStatus
from app.models.extraction import ExtractionStatus
from app.schemas.requests import ExtractionStrategy
from app.services.document_processor import DocumentProcessor
from app.services.page_extraction import PageStrategyRunner
from app.services.tool_call_extractor import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT_TEMPLATE,
    ToolCallExtractor,
)


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
def tool_extractor(mock_llm_client: AsyncMock) -> ToolCallExtractor:
    return ToolCallExtractor(
        llm_client=mock_llm_client,
        model="test-model",
    )


@pytest.fixture
def page_runner(tool_extractor: ToolCallExtractor) -> PageStrategyRunner:
    return PageStrategyRunner(text_extractor=tool_extractor)


@pytest.fixture
def document_processor(
    page_runner: PageStrategyRunner,
) -> DocumentProcessor:
    return _build_document_processor(
        classifier=AsyncMock(),
        page_runner=page_runner,
    )


def _build_document_processor(
    *,
    classifier: AsyncMock,
    page_runner: PageStrategyRunner,
    db: AsyncMock | None = None,
) -> DocumentProcessor:
    db = db or AsyncMock()
    db.add = MagicMock()
    return DocumentProcessor(
        classifier=classifier,
        page_runner=page_runner,
    )


@pytest.mark.asyncio
async def test_extract_from_text_success_first_try(
    tool_extractor: ToolCallExtractor,
    mock_llm_client: AsyncMock,
) -> None:
    expected_data = {"vendor_name": "Acme Corp", "total_amount": 100.0}
    mock_llm_client.call_tool.return_value = ToolCallResult(
        arguments=expected_data,
        input_tokens=100,
        output_tokens=50,
    )

    result, input_tokens, output_tokens = (
        await tool_extractor.extract(
            text="Acme Corp invoice...", doc_type="invoice"
        )
    )

    assert result == expected_data
    assert input_tokens == 100
    assert output_tokens == 50
    mock_llm_client.call_tool.assert_called_once()


@pytest.mark.asyncio
async def test_extract_from_text_success_after_retry(
    tool_extractor: ToolCallExtractor,
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
        await tool_extractor.extract(
            text="Acme Corp invoice...", doc_type="invoice"
        )
    )

    assert result == expected_data
    assert input_tokens == 150
    assert output_tokens == 50
    assert mock_llm_client.call_tool.call_count == 2


@pytest.mark.asyncio
async def test_extract_from_text_counts_tokens_from_no_tool_retry(
    tool_extractor: ToolCallExtractor,
    mock_llm_client: AsyncMock,
) -> None:
    expected_data = {"vendor_name": "Acme Corp"}
    mock_llm_client.call_tool.side_effect = [
        ToolCallResult(arguments=None, input_tokens=20, output_tokens=5),
        ToolCallResult(
            arguments=expected_data,
            input_tokens=150,
            output_tokens=50,
        ),
    ]

    result, input_tokens, output_tokens = (
        await tool_extractor.extract(
            text="Acme Corp invoice...", doc_type="invoice"
        )
    )

    assert result == expected_data
    assert input_tokens == 170
    assert output_tokens == 55


@pytest.mark.asyncio
async def test_extract_from_text_exhausted_retries(
    mock_llm_client: AsyncMock,
) -> None:
    mock_llm_client.call_tool.return_value = None
    service = ToolCallExtractor(
        llm_client=mock_llm_client,
        max_retries=2,
    )

    with pytest.raises(ExtractionError) as exc_info:
        await service.extract(
            text="Acme Corp invoice...", doc_type="invoice"
        )

    assert exc_info.value.attempts == 3
    assert "after 3 attempts" in str(exc_info.value)
    assert mock_llm_client.call_tool.call_count == 3


@pytest.mark.asyncio
async def test_extract_from_text_passes_tool_inputs(
    tool_extractor: ToolCallExtractor,
    mock_llm_client: AsyncMock,
) -> None:
    mock_llm_client.call_tool.return_value = ToolCallResult(
        arguments={"vendor_name": "Acme Corp"},
        input_tokens=100,
        output_tokens=50,
    )

    await tool_extractor.extract(
        text="Acme Corp invoice...", doc_type="invoice"
    )

    call = mock_llm_client.call_tool.call_args
    assert call.kwargs["model"] == "test-model"
    assert call.kwargs["tool_name"] == "extract_invoice"
    assert call.kwargs["tool"]["function"]["name"] == "extract_invoice"
    assert call.kwargs["system_prompt"] == EXTRACTION_SYSTEM_PROMPT
    assert call.kwargs["user_content"] == (
        EXTRACTION_USER_PROMPT_TEMPLATE.format(text="Acme Corp invoice...")
    )
    assert "Normalize dates to ISO 8601" in call.kwargs["user_content"]
    assert "Do not guess missing values" in call.kwargs["user_content"]


@pytest.mark.asyncio
async def test_extract_from_text_wraps_llm_client_error(
    tool_extractor: ToolCallExtractor,
    mock_llm_client: AsyncMock,
) -> None:
    mock_llm_client.call_tool.side_effect = LLMClientError("bad response")

    with pytest.raises(ExtractionError) as exc_info:
        await tool_extractor.extract(
            text="Acme Corp invoice...", doc_type="invoice"
        )

    assert exc_info.value.attempts == 1
    assert isinstance(exc_info.value.original_exception, LLMClientError)


@pytest.mark.asyncio
async def test_extract_from_pages_full_uses_all_pages(
    page_runner: PageStrategyRunner,
    tool_extractor: ToolCallExtractor,
) -> None:
    tool_extractor.extract = AsyncMock(
        return_value=(
            {"vendor_name": "Acme Corp"},
            100,
            50,
        )
    )

    result = await page_runner.extract_from_pages(
        pages=["First page", "Second page"],
        doc_type="invoice",
        strategy=ExtractionStrategy.FULL,
    )

    assert result.strategy == ExtractionStrategy.FULL
    assert result.source_pages == [0, 1]
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    call = tool_extractor.extract.call_args
    assert "Page 0:\nFirst page" in call.kwargs["text"]
    assert "Page 1:\nSecond page" in call.kwargs["text"]


def test_resolve_strategy_defaults_by_page_count() -> None:
    assert (
        PageStrategyRunner.resolve_strategy(strategy=None, page_count=10)
        == ExtractionStrategy.FULL
    )
    assert (
        PageStrategyRunner.resolve_strategy(strategy=None, page_count=11)
        == ExtractionStrategy.SMART
    )
    assert (
        PageStrategyRunner.resolve_strategy(
            strategy="smart",
            page_count=1,
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

    selected = PageStrategyRunner.select_pages(pages)

    assert selected == [
        (0, "Cover"),
        (2, "Total amount due is USD 50"),
        (3, "Signature"),
    ]


@pytest.mark.asyncio
async def test_page_by_page_merges_schema_fields_and_tracks_sources(
    page_runner: PageStrategyRunner,
    tool_extractor: ToolCallExtractor,
) -> None:
    tool_extractor.extract = AsyncMock(
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

    result = await page_runner.extract_from_pages(
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
async def test_page_by_page_preserves_scalar_source_pages_on_better_value(
    page_runner: PageStrategyRunner,
    tool_extractor: ToolCallExtractor,
) -> None:
    tool_extractor.extract = AsyncMock(
        side_effect=[
            ({"governing_law": "NY"}, 10, 5),
            ({"governing_law": "New York State"}, 12, 6),
        ]
    )

    result = await page_runner.extract_from_pages(
        pages=["Short law", "Specific law"],
        doc_type="contract",
        strategy=ExtractionStrategy.PAGE_BY_PAGE,
    )

    assert result.raw_output == {"governing_law": "New York State"}
    assert result.source_pages == [0, 1]
    assert result.field_source_pages == {"governing_law": [0, 1]}


@pytest.mark.asyncio
async def test_page_by_page_skips_failed_pages_when_others_succeed(
    page_runner: PageStrategyRunner,
    tool_extractor: ToolCallExtractor,
) -> None:
    tool_extractor.extract = AsyncMock(
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

    result = await page_runner.extract_from_pages(
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
    tool_extractor = ToolCallExtractor(
        llm_client=mock_llm_client,
        max_retries=0,
    )
    page_runner = PageStrategyRunner(text_extractor=tool_extractor)
    service = _build_document_processor(
        classifier=AsyncMock(),
        page_runner=page_runner,
    )
    mock_llm_client.call_tool.return_value = None

    with pytest.raises(Exception) as exc_info:
        await service.process_pages(
            pages=["Some contract data"],
            page_count=1,
            doc_type="contract",
        )

    assert isinstance(exc_info.value.original, ExtractionError)
    assert exc_info.value.resolved.doc_type == "contract"


@pytest.mark.asyncio
async def test_extract_upload_records_tokens_from_extraction_error(
) -> None:
    tool_extractor = AsyncMock()
    tool_extractor.extract.side_effect = ExtractionError(
        "LLM failed",
        attempts=2,
        input_tokens=123,
        output_tokens=45,
    )
    service = _build_document_processor(
        classifier=AsyncMock(),
        page_runner=PageStrategyRunner(text_extractor=tool_extractor),
    )

    with pytest.raises(Exception) as exc_info:
        await service.process_pages(
            pages=["Some contract data"],
            page_count=1,
            doc_type="contract",
        )

    assert exc_info.value.original.input_tokens == 123
    assert exc_info.value.original.output_tokens == 45


@pytest.mark.asyncio
async def test_extract_upload_persists_failed_validation_result(
    mock_llm_client: AsyncMock,
) -> None:
    tool_extractor = ToolCallExtractor(llm_client=mock_llm_client)
    service = _build_document_processor(
        classifier=AsyncMock(),
        page_runner=PageStrategyRunner(text_extractor=tool_extractor),
    )
    mock_llm_client.call_tool.return_value = ToolCallResult(
        arguments={"invalid_field": "some data"},
        input_tokens=50,
        output_tokens=10,
    )

    response = await service.process_pages(
        pages=["Some contract data"],
        page_count=1,
        doc_type="contract",
    )

    assert response.pipeline.validation.status == ExtractionStatus.FAILED
    assert response.pipeline.extraction.audit_output == {
        "invalid_field": "some data"
    }
    assert response.pipeline.warnings
    assert response.pipeline.validation.extracted_data is None


@pytest.mark.asyncio
async def test_extract_upload_fails_when_audit_persist_fails(
    mock_llm_client: AsyncMock,
) -> None:
    tool_extractor = ToolCallExtractor(llm_client=mock_llm_client)
    service = _build_document_processor(
        classifier=AsyncMock(),
        page_runner=PageStrategyRunner(text_extractor=tool_extractor),
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

    result = await service.process_pages(
        pages=["Some contract data"],
        page_count=1,
        doc_type="contract",
    )
    assert result.pipeline.validation.status == ExtractionStatus.COMPLETED


@pytest.mark.asyncio
async def test_extract_upload_validates_explicit_doc_type_before_side_effects(
) -> None:
    document_processor = _build_document_processor(
        classifier=AsyncMock(),
        page_runner=PageStrategyRunner(text_extractor=AsyncMock()),
    )
    with pytest.raises(UnsupportedDocumentTypeError) as exc_info:
        await document_processor.process_pages(
            pages=["Some data"],
            page_count=1,
            doc_type="banana",
        )

    assert "Unsupported document type: banana" in str(exc_info.value)


@pytest.mark.asyncio
async def test_extract_upload_rejects_zero_page_pdf_before_side_effects(
) -> None:
    from app.services.document_intake import DocumentIntakeService

    storage = AsyncMock()
    intake = DocumentIntakeService(storage=storage)

    with pytest.raises(PDFParseError) as exc_info:
        await intake.prepare_document(
            content=_create_zero_page_pdf(),
            filename="empty.pdf",
            status=DocumentStatus.UPLOADED,
        )

    assert "PDF does not contain any pages" in str(exc_info.value)
    storage.save.assert_not_called()


@pytest.mark.asyncio
async def test_extract_upload_offloads_pdf_parsing(
    mock_llm_client: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = AsyncMock()
    storage.save.return_value = "test-id/test.pdf"
    from app.services.document_intake import DocumentIntakeService

    intake = DocumentIntakeService(storage=storage)

    run_sync = AsyncMock()

    async def fake_run_sync(func, *args):
        return func(*args)

    run_sync.side_effect = fake_run_sync
    monkeypatch.setattr(
        "app.services.document_intake.anyio.to_thread.run_sync", run_sync
    )

    await intake.prepare_document(
        content=_create_test_pdf("Some contract data"),
        filename="test.pdf",
        status=DocumentStatus.UPLOADED,
    )

    run_sync.assert_called_once()


@pytest.mark.asyncio
async def test_document_intake_returns_document_and_pages(
) -> None:
    from app.services.document_intake import DocumentIntakeService

    storage = AsyncMock()
    storage.save.return_value = "test-id/test.pdf"
    intake = DocumentIntakeService(storage=storage)

    result = await intake.prepare_document(
        content=_create_test_pdf("Invoice page"),
        filename="invoice.pdf",
        status=DocumentStatus.UPLOADED,
    )

    assert result.document.filename == "invoice.pdf"
    assert result.document.file_path == "test-id/test.pdf"
    assert result.document.status == DocumentStatus.UPLOADED
    assert result.pages_text


@pytest.mark.asyncio
async def test_extract_existing_document_raises_not_found_error(
) -> None:
    classifier = AsyncMock()
    classifier.classify.return_value.doc_type = "invoice"
    classifier.classify.return_value.usage = MagicMock()
    page_runner = MagicMock()
    page_runner.resolve_strategy.return_value = ExtractionStrategy.FULL
    page_runner.extract_from_pages = AsyncMock(
        return_value=MagicMock(
        raw_output={"vendor_name": "Acme"},
        input_tokens=1,
        output_tokens=1,
        strategy=ExtractionStrategy.FULL,
        source_pages=[0],
        warnings=[],
        audit_output={"vendor_name": "Acme"},
        )
    )
    processor = _build_document_processor(
        classifier=classifier,
        page_runner=page_runner,
    )

    await processor.process_pages(
        pages=["Invoice text"],
        page_count=1,
        doc_type=None,
    )

    classifier.classify.assert_awaited_once_with("Invoice text")
