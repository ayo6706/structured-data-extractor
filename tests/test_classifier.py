from unittest.mock import ANY, AsyncMock

import pytest

from app.core.exceptions import ClassificationError
from app.integrations.llm.client import LLMClientError, TextCompletionResult
from app.services.classifier import ClassifierService


def _create_service(content: str = "") -> tuple[ClassifierService, AsyncMock]:
    llm_client = AsyncMock()
    llm_client.generate_text.return_value = TextCompletionResult(
        content=content,
        input_tokens=10,
        output_tokens=2,
    )
    return ClassifierService(llm_client=llm_client), llm_client.generate_text


@pytest.mark.asyncio
async def test_classify_invoice() -> None:
    service, complete = _create_service("invoice")
    result = await service.classify("Invoice detail text")

    assert result.doc_type == "invoice"
    complete.assert_called_once_with(
        model=service.model,
        messages=[{"role": "user", "content": ANY}],
        max_tokens=25,
    )


@pytest.mark.asyncio
async def test_classify_contract() -> None:
    service, _ = _create_service("contract")
    result = await service.classify("Agreement contract text")

    assert result.doc_type == "contract"


@pytest.mark.asyncio
async def test_classify_payslip() -> None:
    service, _ = _create_service("payslip")
    result = await service.classify("Earnings statement payslip")

    assert result.doc_type == "payslip"


@pytest.mark.asyncio
async def test_classify_receipt() -> None:
    service, _ = _create_service("receipt")
    result = await service.classify("Store purchase receipt")

    assert result.doc_type == "receipt"


@pytest.mark.asyncio
async def test_classify_rejects_prose_response() -> None:
    service, _ = _create_service("It seems to be a Contract document.")
    result = await service.classify("Contract text")

    assert result.doc_type == "unknown"


@pytest.mark.asyncio
async def test_classify_accepts_case_and_whitespace_variation() -> None:
    service, _ = _create_service(" Contract\n")
    result = await service.classify("Contract text")

    assert result.doc_type == "contract"


@pytest.mark.asyncio
async def test_classify_unknown() -> None:
    service, _ = _create_service("unknown")
    result = await service.classify("Some unknown document content")

    assert result.doc_type == "unknown"


@pytest.mark.asyncio
async def test_classify_prompt_allows_unknown() -> None:
    service, complete = _create_service("unknown")

    await service.classify("Some unsupported document content")

    prompt = complete.call_args.kwargs["messages"][0]["content"]
    assert "invoice, contract, payslip, receipt, unknown" in prompt
    assert "Use unknown when" in prompt


@pytest.mark.asyncio
async def test_classify_garbage_response_returns_unknown() -> None:
    service, _ = _create_service("banana")
    result = await service.classify("Some unknown document content")

    assert result.doc_type == "unknown"


@pytest.mark.asyncio
async def test_classify_empty() -> None:
    service, _ = _create_service("")
    result = await service.classify("Empty doc")

    assert result.doc_type == "unknown"


@pytest.mark.asyncio
async def test_classify_llm_failure() -> None:
    service, complete = _create_service()
    complete.side_effect = LLMClientError("API connection timed out")

    with pytest.raises(ClassificationError) as exc_info:
        await service.classify("Sample text")

    assert "LLM classification failed" in str(exc_info.value)
    assert isinstance(exc_info.value.original_exception, LLMClientError)


@pytest.mark.asyncio
async def test_classify_limits_prompt_to_first_500_characters() -> None:
    service, complete = _create_service("invoice")
    text = "a" * 500 + "TAIL_MARKER"

    await service.classify(text)

    prompt = complete.call_args.kwargs["messages"][0]["content"]
    assert "a" * 500 in prompt
    assert "TAIL_MARKER" not in prompt


@pytest.mark.asyncio
async def test_classify_does_not_match_doc_type_inside_other_words() -> None:
    service, _ = _create_service("invoiceless")
    result = await service.classify("Sample text")

    assert result.doc_type == "unknown"


@pytest.mark.asyncio
async def test_classify_preserves_registered_type_casing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.classifier.SchemaRegistry.list_types",
        lambda: ["Invoice"],
    )
    service, _ = _create_service("invoice")
    result = await service.classify("Invoice detail text")

    assert result.doc_type == "Invoice"


@pytest.mark.asyncio
async def test_classify_returns_classifier_token_usage() -> None:
    service, _ = _create_service("invoice")

    result = await service.classify("Invoice detail text")

    assert result.doc_type == "invoice"
    assert result.usage.model == service.model
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 2
