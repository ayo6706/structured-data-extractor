import logging
from typing import Final

from app.core.config import llm_settings
from app.core.exceptions import ExtractionError
from app.integrations.llm.client import LLMClient, LLMClientError
from app.schemas.registry import SchemaRegistry

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT: Final[str] = (
    "You are a document data extraction assistant. "
    "Your job is to extract structured data from the document text. "
    "You must call the extraction tool. Do not provide prose explanations."
)

RETRY_USER_PROMPT: Final[str] = (
    "You must respond ONLY by calling the specified tool. "
    "Do not output prose or conversational replies."
)

EXTRACTION_USER_PROMPT_TEMPLATE: Final[str] = """
Extract all structured data from this document.

Rules:
- Preserve the exact numeric amounts stated in the document.
- Normalize dates to ISO 8601 format (YYYY-MM-DD) when the date is stated.
- Do not guess missing values.
- Omit fields that are genuinely absent from the document.
- Respond only by calling the extraction tool.

Document text:
{text}
""".strip()


class ToolCallExtractor:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        model: str = llm_settings.EXTRACTION_MODEL,
        max_retries: int = llm_settings.MAX_RETRIES,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.max_retries = max_retries

    async def extract(
        self, *, text: str, doc_type: str
    ) -> tuple[dict, int, int]:
        tool = SchemaRegistry.get_tool(doc_type)
        tool_name = f"extract_{doc_type}"
        total_input_tokens = 0
        total_output_tokens = 0
        attempts = 0
        extra_messages: list[dict[str, str]] = []

        while attempts <= self.max_retries:
            attempts += 1
            try:
                result = await self.llm_client.call_tool(
                    model=self.model,
                    system_prompt=EXTRACTION_SYSTEM_PROMPT,
                    user_content=EXTRACTION_USER_PROMPT_TEMPLATE.format(
                        text=text
                    ),
                    tool=tool,
                    tool_name=tool_name,
                    extra_messages=extra_messages or None,
                )
            except LLMClientError as exc:
                raise ExtractionError(
                    message="LLM tool-call extraction failed",
                    attempts=attempts,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    original_exception=exc,
                ) from exc

            if result is None or result.arguments is None:
                if result is not None:
                    total_input_tokens += result.input_tokens
                    total_output_tokens += result.output_tokens
                logger.warning(
                    "LLM did not return a tool call on attempt %d",
                    attempts,
                )
                if attempts <= self.max_retries:
                    extra_messages.append({"role": "assistant", "content": ""})
                    extra_messages.append(
                        {"role": "user", "content": RETRY_USER_PROMPT}
                    )
                    continue
                break

            total_input_tokens += result.input_tokens
            total_output_tokens += result.output_tokens
            return result.arguments, total_input_tokens, total_output_tokens

        raise ExtractionError(
            message=(
                f"Failed to extract structured data for '{doc_type}' "
                f"after {attempts} attempts. LLM failed to call tool."
            ),
            attempts=attempts,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )
