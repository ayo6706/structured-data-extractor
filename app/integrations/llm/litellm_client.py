import litellm

from app.integrations.llm.client import LLMClient, LLMClientError


class LiteLLMClient(LLMClient):
    async def complete(
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
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMClientError(
                "LLM completion failed",
                original_exception=exc,
            ) from exc
