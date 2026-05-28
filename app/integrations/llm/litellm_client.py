import json
from typing import Any

import litellm

from app.integrations.llm.client import (
    LLMClient,
    LLMClientError,
    ToolCallResult,
)


class LiteLLMClient(LLMClient):
    async def generate_text(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> str:
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            refusal = getattr(choice.message, "refusal", None)
            if isinstance(refusal, str) and refusal:
                raise LLMClientError(
                    "LLM completion failed: request was refused. "
                    f"Reason: {refusal}"
                )
            return choice.message.content or ""
        except LLMClientError:
            raise
        except Exception as exc:
            raise LLMClientError(
                "LLM completion failed",
                original_exception=exc,
            ) from exc

    async def call_tool(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: str,
        tool: dict[str, Any],
        tool_name: str,
        extra_messages: list[dict[str, str]] | None = None,
    ) -> ToolCallResult | None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        if extra_messages:
            messages.extend(extra_messages)

        tool_choice = {"type": "function", "function": {"name": tool_name}}

        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                tools=[tool],
                tool_choice=tool_choice,
            )
        except Exception as exc:
            raise LLMClientError(
                "LLM completion failed",
                original_exception=exc,
            ) from exc

        try:
            choice = response.choices[0]
            message = choice.message
        except (IndexError, AttributeError) as exc:
            raise LLMClientError(
                "LLM completion failed: invalid response structure",
                original_exception=exc,
            ) from exc

        refusal = getattr(message, "refusal", None)
        if isinstance(refusal, str) and refusal:
            raise LLMClientError(
                f"LLM completion failed: request was refused. Reason: {refusal}"
            )

        usage = getattr(response, "usage", None)
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return None

        try:
            tool_call = tool_calls[0]
            args_val = tool_call.function.arguments
            if isinstance(args_val, dict):
                arguments = args_val
            else:
                arguments = json.loads(args_val)
            return ToolCallResult(
                arguments=arguments,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as exc:
            raise LLMClientError(
                "Failed to parse tool call arguments",
                original_exception=exc,
            ) from exc
