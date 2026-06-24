from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.integrations.llm.client import LLMClientError
from app.integrations.llm.litellm_client import LiteLLMClient

LLM_CLIENT_ERROR_MESSAGE = "LLM completion failed"
TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_invoice",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _create_mock_response(content: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.message.refusal = None
    response.choices = [choice]
    response.usage.prompt_tokens = 12
    response.usage.completion_tokens = 3
    return response


@pytest.mark.asyncio
@patch("app.integrations.llm.litellm_client.litellm.acompletion")
async def test_litellm_client_generate_text_returns_text(
    mock_acompletion: AsyncMock,
) -> None:
    mock_acompletion.return_value = _create_mock_response("invoice")
    client = LiteLLMClient()

    result = await client.generate_text(
        model="test-model",
        messages=[{"role": "user", "content": "Classify this"}],
        max_tokens=10,
    )

    assert result.content == "invoice"
    assert result.input_tokens == 12
    assert result.output_tokens == 3
    mock_acompletion.assert_called_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "Classify this"}],
        max_tokens=10,
    )


@pytest.mark.asyncio
@patch("app.integrations.llm.litellm_client.litellm.acompletion")
async def test_litellm_client_wraps_provider_errors(
    mock_acompletion: AsyncMock,
) -> None:
    mock_acompletion.side_effect = Exception("timeout")
    client = LiteLLMClient()

    with pytest.raises(LLMClientError) as exc_info:
        await client.generate_text(
            model="test-model",
            messages=[{"role": "user", "content": "Classify this"}],
            max_tokens=10,
        )

    assert str(exc_info.value) == LLM_CLIENT_ERROR_MESSAGE


@pytest.mark.asyncio
@patch("app.integrations.llm.litellm_client.litellm.acompletion")
async def test_litellm_client_wraps_unexpected_response_shape(
    mock_acompletion: AsyncMock,
) -> None:
    response = MagicMock()
    response.choices = []
    mock_acompletion.return_value = response
    client = LiteLLMClient()

    with pytest.raises(LLMClientError) as exc_info:
        await client.generate_text(
            model="test-model",
            messages=[{"role": "user", "content": "Classify this"}],
            max_tokens=10,
        )

    assert str(exc_info.value) == LLM_CLIENT_ERROR_MESSAGE


@pytest.mark.asyncio
@patch("app.integrations.llm.litellm_client.litellm.acompletion")
async def test_litellm_client_call_tool_success(
    mock_acompletion: AsyncMock,
) -> None:
    mock_response = MagicMock()
    choice = MagicMock()
    tool_call = MagicMock()
    tool_call.function.name = "extract_invoice"
    tool_call.function.arguments = '{"vendor_name": "Acme Corp"}'
    choice.message.tool_calls = [tool_call]
    choice.message.refusal = None
    mock_response.choices = [choice]
    mock_response.usage.prompt_tokens = 80
    mock_response.usage.completion_tokens = 40
    mock_acompletion.return_value = mock_response

    client = LiteLLMClient()
    result = await client.call_tool(
        model="test-model",
        system_prompt="Extract data.",
        user_content="Document text",
        tool=TEST_TOOL,
        tool_name="extract_invoice",
    )

    assert result is not None
    assert result.arguments == {"vendor_name": "Acme Corp"}
    assert result.input_tokens == 80
    assert result.output_tokens == 40
    mock_acompletion.assert_called_once_with(
        model="test-model",
        messages=[
            {"role": "system", "content": "Extract data."},
            {"role": "user", "content": "Document text"},
        ],
        tools=[TEST_TOOL],
        tool_choice={
            "type": "function",
            "function": {"name": "extract_invoice"},
        },
    )


@pytest.mark.asyncio
@patch("app.integrations.llm.litellm_client.litellm.acompletion")
async def test_litellm_client_call_tool_accepts_dict_arguments(
    mock_acompletion: AsyncMock,
) -> None:
    mock_response = MagicMock()
    choice = MagicMock()
    tool_call = MagicMock()
    tool_call.function.name = "extract_invoice"
    tool_call.function.arguments = {"vendor_name": "Acme Corp"}
    choice.message.tool_calls = [tool_call]
    choice.message.refusal = None
    mock_response.choices = [choice]
    mock_response.usage.prompt_tokens = 80
    mock_response.usage.completion_tokens = 40
    mock_acompletion.return_value = mock_response

    client = LiteLLMClient()
    result = await client.call_tool(
        model="test-model",
        system_prompt="Extract data.",
        user_content="Document text",
        tool=TEST_TOOL,
        tool_name="extract_invoice",
    )

    assert result is not None
    assert result.arguments == {"vendor_name": "Acme Corp"}
    assert result.input_tokens == 80
    assert result.output_tokens == 40


@pytest.mark.asyncio
@patch("app.integrations.llm.litellm_client.litellm.acompletion")
async def test_litellm_client_rejects_unexpected_tool_name(
    mock_acompletion: AsyncMock,
) -> None:
    mock_response = MagicMock()
    choice = MagicMock()
    tool_call = MagicMock()
    tool_call.function.name = "extract_contract"
    tool_call.function.arguments = '{"vendor_name": "Acme Corp"}'
    choice.message.tool_calls = [tool_call]
    choice.message.refusal = None
    mock_response.choices = [choice]
    mock_response.usage.prompt_tokens = 80
    mock_response.usage.completion_tokens = 40
    mock_acompletion.return_value = mock_response

    client = LiteLLMClient()
    with pytest.raises(LLMClientError) as exc_info:
        await client.call_tool(
            model="test-model",
            system_prompt="Extract data.",
            user_content="Document text",
            tool=TEST_TOOL,
            tool_name="extract_invoice",
        )

    assert (
        "Tool name mismatch: expected tool call "
        "'extract_invoice', got 'extract_contract'"
    ) in str(exc_info.value)


@pytest.mark.asyncio
@patch("app.integrations.llm.litellm_client.litellm.acompletion")
async def test_litellm_client_call_tool_prose(
    mock_acompletion: AsyncMock,
) -> None:
    mock_response = MagicMock()
    choice = MagicMock()
    choice.message.tool_calls = None
    choice.message.content = "No tool call here"
    choice.message.refusal = None
    mock_response.choices = [choice]
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 10
    mock_acompletion.return_value = mock_response

    client = LiteLLMClient()
    result = await client.call_tool(
        model="test-model",
        system_prompt="Extract data.",
        user_content="Document text",
        tool=TEST_TOOL,
        tool_name="extract_invoice",
        extra_messages=[{"role": "user", "content": "Try again"}],
    )

    assert result is not None
    assert result.arguments is None
    assert result.input_tokens == 50
    assert result.output_tokens == 10
    mock_acompletion.assert_called_once_with(
        model="test-model",
        messages=[
            {"role": "system", "content": "Extract data."},
            {"role": "user", "content": "Document text"},
            {"role": "user", "content": "Try again"},
        ],
        tools=[TEST_TOOL],
        tool_choice={
            "type": "function",
            "function": {"name": "extract_invoice"},
        },
    )


@pytest.mark.asyncio
@patch("app.integrations.llm.litellm_client.litellm.acompletion")
async def test_litellm_client_call_tool_refusal(
    mock_acompletion: AsyncMock,
) -> None:
    mock_response = MagicMock()
    choice = MagicMock()
    choice.message.refusal = "I cannot process this request."
    mock_response.choices = [choice]
    mock_acompletion.return_value = mock_response

    client = LiteLLMClient()
    with pytest.raises(LLMClientError) as exc_info:
        await client.call_tool(
            model="test-model",
            system_prompt="Extract data.",
            user_content="Document text",
            tool=TEST_TOOL,
            tool_name="extract_invoice",
        )

    assert "request was refused" in str(exc_info.value)
