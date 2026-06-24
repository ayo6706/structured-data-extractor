import logging
from typing import Final

from app.core.config import llm_settings
from app.core.exceptions import ClassificationError
from app.integrations.llm.client import LLMClient, LLMClientError
from app.models.llm_usage import LLMUsagePurpose
from app.schemas.registry import SchemaRegistry
from app.services.types import ClassificationResult, LLMUsageEvent

logger = logging.getLogger(__name__)

CLASSIFIER_PROMPT: Final[str] = (
    "You are a document classifier. Based on the document text below, "
    "identify the document type.\n"
    "Respond with exactly one of: {valid_types}\n\n"
    "Use unknown when the document does not match one of those types.\n\n"
    "Document text (first 500 chars):\n{text}\n\n"
    "Document type:"
)
TEXT_PREVIEW_LENGTH: Final[int] = 500
UNKNOWN_DOC_TYPE: Final[str] = "unknown"


class ClassifierService:
    def __init__(
        self,
        llm_client: LLMClient,
        model: str = llm_settings.CLASSIFIER_MODEL,
    ) -> None:
        self.llm_client = llm_client
        self.model = model

    async def classify(self, first_page_text: str) -> ClassificationResult:
        valid_types = SchemaRegistry.list_types()
        valid_type_map = {
            doc_type.lower(): doc_type for doc_type in valid_types
        }
        classifier_outputs = [*valid_types, UNKNOWN_DOC_TYPE]
        prompt = CLASSIFIER_PROMPT.format(
            valid_types=", ".join(classifier_outputs),
            text=first_page_text[:TEXT_PREVIEW_LENGTH],
        )

        try:
            completion = await self.llm_client.generate_text(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=25,
            )
        except LLMClientError as exc:
            raise ClassificationError(
                f"LLM classification failed: {exc}",
                original_exception=exc,
            ) from exc

        usage = LLMUsageEvent(
            purpose=LLMUsagePurpose.CLASSIFIER,
            model=self.model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )
        normalized = completion.content.strip().lower()

        if normalized == UNKNOWN_DOC_TYPE:
            return ClassificationResult(doc_type=UNKNOWN_DOC_TYPE, usage=usage)

        if normalized in valid_type_map:
            return ClassificationResult(
                doc_type=valid_type_map[normalized],
                usage=usage,
            )

        return ClassificationResult(doc_type=UNKNOWN_DOC_TYPE, usage=usage)
