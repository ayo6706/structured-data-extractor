import os

import pytest

from app.integrations.llm.litellm_client import LiteLLMClient

RUN_LIVE_LLM_SMOKE = os.getenv("RUN_LIVE_LLM_SMOKE") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_LIVE_LLM_SMOKE,
    reason="RUN_LIVE_LLM_SMOKE=1 is required for live LLM smoke tests",
)

TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_invoice",
        "description": "Extract invoice smoke-test fields",
        "parameters": {
            "type": "object",
            "properties": {
                "vendor_name": {
                    "type": "string",
                    "description": "Vendor name stated on the invoice",
                },
                "invoice_number": {
                    "type": "string",
                    "description": "Invoice number stated on the invoice",
                },
            },
            "required": ["vendor_name", "invoice_number"],
        },
    },
}

PROVIDER_CASES = [
    (
        "gemini",
        "GEMINI_API_KEY",
        os.getenv("LIVE_GEMINI_MODEL", "gemini/gemini-2.5-flash"),
    ),
    (
        "anthropic",
        "ANTHROPIC_API_KEY",
        os.getenv("LIVE_ANTHROPIC_MODEL", "anthropic/claude-3-5-haiku"),
    ),
    (
        "openai",
        "OPENAI_API_KEY",
        os.getenv("LIVE_OPENAI_MODEL", "openai/gpt-4.1-mini"),
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "api_key_env", "model"), PROVIDER_CASES)
async def test_live_provider_forced_tool_call(
    provider: str,
    api_key_env: str,
    model: str,
) -> None:
    if not os.getenv(api_key_env):
        pytest.skip(f"{api_key_env} is not configured")

    client = LiteLLMClient()

    result = await client.call_tool(
        model=model,
        system_prompt=(
            "Extract only the requested invoice fields. "
            "You must call the tool."
        ),
        user_content=(
            "Invoice\n"
            "Vendor: Acme Corp\n"
            "Invoice Number: INV-SMOKE-001\n"
            "Total: USD 10.00\n"
            f"Provider smoke case: {provider}"
        ),
        tool=TEST_TOOL,
        tool_name="extract_invoice",
    )

    assert result is not None
    assert result.arguments is not None
    assert isinstance(result.arguments.get("vendor_name"), str)
    assert result.arguments["vendor_name"].strip()
    assert isinstance(result.arguments.get("invoice_number"), str)
    assert result.arguments["invoice_number"].strip()
