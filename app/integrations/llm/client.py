from abc import ABC, abstractmethod


class LLMClientError(Exception):
    def __init__(
        self, message: str, original_exception: Exception | None = None
    ) -> None:
        self.original_exception = original_exception
        super().__init__(message)


class LLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> str:
        pass
