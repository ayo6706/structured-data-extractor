from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.integrations.llm.client import LLMClientError
from app.integrations.llm.litellm_client import LiteLLMClient

TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_invoice",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _tool_response(
    *,
    tool_name: str = "extract_invoice",
    arguments='{"vendor_name": "Acme Corp"}',
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
):
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name=tool_name, arguments=arguments)
    )
    message = SimpleNamespace(tool_calls=[tool_call], refusal=None)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "model", "arguments"),
    [
        ("openai", "openai/gpt-4.1-mini", '{"vendor_name": "Acme Corp"}'),
        ("anthropic", "anthropic/claude-3-5-haiku", {"vendor_name": "Acme"}),
        ("gemini", "gemini/gemini-2.5-flash", '{"vendor_name": "Gemini"}'),
    ],
)
@patch("app.integrations.llm.litellm_client.litellm.acompletion")
async def test_litellm_provider_tool_contract_fixtures(
    mock_acompletion: AsyncMock,
    provider: str,
    model: str,
    arguments,
) -> None:
    mock_acompletion.return_value = _tool_response(arguments=arguments)
    client = LiteLLMClient()

    result = await client.call_tool(
        model=model,
        system_prompt="Extract data.",
        user_content=f"{provider} fixture",
        tool=TEST_TOOL,
        tool_name="extract_invoice",
    )

    assert result is not None
    assert result.arguments is not None
    assert "vendor_name" in result.arguments
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    mock_acompletion.assert_called_once()
    call = mock_acompletion.call_args.kwargs
    assert call["model"] == model
    assert call["tools"] == [TEST_TOOL]
    if "gemini" in model.lower():
        assert call["tool_choice"] == "auto"
    else:
        assert call["tool_choice"] == {
            "type": "function",
            "function": {"name": "extract_invoice"},
        }


@pytest.mark.asyncio
@patch("app.integrations.llm.litellm_client.litellm.acompletion")
async def test_provider_contract_rejects_wrong_tool_name(
    mock_acompletion: AsyncMock,
) -> None:
    mock_acompletion.return_value = _tool_response(tool_name="extract_receipt")
    client = LiteLLMClient()

    with pytest.raises(LLMClientError):
        await client.call_tool(
            model="anthropic/claude-3-5-haiku",
            system_prompt="Extract data.",
            user_content="bad fixture",
            tool=TEST_TOOL,
            tool_name="extract_invoice",
        )
