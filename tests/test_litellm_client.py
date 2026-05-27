from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.integrations.llm.client import LLMClientError
from app.integrations.llm.litellm_client import LiteLLMClient

LLM_CLIENT_ERROR_MESSAGE = "LLM completion failed"


def _create_mock_response(content: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    response.choices = [choice]
    return response


@pytest.mark.asyncio
@patch("app.integrations.llm.litellm_client.litellm.acompletion")
async def test_litellm_client_returns_completion_text(
    mock_acompletion: AsyncMock,
) -> None:
    mock_acompletion.return_value = _create_mock_response("invoice")
    client = LiteLLMClient()

    result = await client.complete(
        model="test-model",
        messages=[{"role": "user", "content": "Classify this"}],
        max_tokens=10,
    )

    assert result == "invoice"
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
        await client.complete(
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
        await client.complete(
            model="test-model",
            messages=[{"role": "user", "content": "Classify this"}],
            max_tokens=10,
        )

    assert str(exc_info.value) == LLM_CLIENT_ERROR_MESSAGE
