from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class LLMClientError(Exception):
    def __init__(
        self, message: str, original_exception: Exception | None = None
    ) -> None:
        self.original_exception = original_exception
        super().__init__(message)


@dataclass
class ToolCallResult:
    arguments: dict[str, Any]
    input_tokens: int
    output_tokens: int


class LLMClient(ABC):
    @abstractmethod
    async def generate_text(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> str:
        pass

    @abstractmethod
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
        """Sends a system and user message pair, forcing a tool call.

        Returns:
            ToolCallResult: Parsed arguments and token usage.
            None: If the model responds with prose (no tool call).

        Raises:
            LLMClientError: If completion or parsing fails.
        """
        pass
