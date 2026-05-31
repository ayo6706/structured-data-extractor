import copy
import json
import logging
from typing import Any

import litellm

from app.integrations.llm.client import (
    LLMClient,
    LLMClientError,
    TextCompletionResult,
    ToolCallResult,
)

logger = logging.getLogger(__name__)


def _dereference_and_simplify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return schema

    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve_and_simplify(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = node["$ref"]
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.split("/")[-1]
                    if def_name in defs:
                        resolved = resolve_and_simplify(defs[def_name])
                        merged = {k: v for k, v in node.items() if k != "$ref"}
                        merged.update(resolved)
                        return merged

            new_node = {}
            for k, v in node.items():
                if k == "anyOf" and isinstance(v, list):
                    subschemas = v
                    types = []
                    for sub in subschemas:
                        if isinstance(sub, dict):
                            t = sub.get("type")
                            if t:
                                types.append(t)

                    non_null_types = [t for t in types if t != "null"]
                    if not non_null_types:
                        primary_type = "string"
                    elif "number" in non_null_types:
                        primary_type = "number"
                    elif "integer" in non_null_types:
                        primary_type = "integer"
                    elif "boolean" in non_null_types:
                        primary_type = "boolean"
                    else:
                        primary_type = non_null_types[0]

                    new_node["type"] = primary_type
                    continue

                new_node[k] = resolve_and_simplify(v)

            for forbidden in ["pattern", "format"]:
                new_node.pop(forbidden, None)

            return new_node

        elif isinstance(node, list):
            return [resolve_and_simplify(item) for item in node]
        return node

    return resolve_and_simplify(schema)


class LiteLLMClient(LLMClient):
    async def generate_text(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> TextCompletionResult:
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
            usage = getattr(response, "usage", None)
            return TextCompletionResult(
                content=choice.message.content or "",
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            )
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

        # Gemini rejects these schema features and returns empty choices when
        # tool_choice is forced.
        if "gemini" in model.lower():
            tool_choice = "auto"
            tool = copy.deepcopy(tool)
            tool["function"]["parameters"] = _dereference_and_simplify_schema(
                tool["function"]["parameters"]
            )
        else:
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
            prompt_feedback = getattr(response, "prompt_feedback", None)
            block_reason = None
            if prompt_feedback:
                block_reason = getattr(
                    prompt_feedback, "block_reason", None
                )
            if block_reason:
                raise LLMClientError(
                    f"LLM completion blocked by provider safety filter: "
                    f"{block_reason}",
                    original_exception=exc,
                ) from exc
            logger.error(
                "LLM returned empty choices. Raw response: %s",
                response,
            )
            raise LLMClientError(
                "LLM completion failed: provider returned no choices "
                "(possible content filter or malformed response)",
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
            return ToolCallResult(
                arguments=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        try:
            tool_call = tool_calls[0]
            returned_name = getattr(tool_call.function, "name", None)
            if returned_name != tool_name:
                raise LLMClientError(
                    "Tool name mismatch: expected tool call "
                    f"'{tool_name}', got '{returned_name}'"
                )
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
        except LLMClientError:
            raise
        except Exception as exc:
            raise LLMClientError(
                "Failed to parse tool call arguments",
                original_exception=exc,
            ) from exc
