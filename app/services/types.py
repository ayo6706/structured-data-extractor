from dataclasses import dataclass, field
from typing import Any

from app.models.llm_usage import LLMUsagePurpose
from app.schemas.requests import ExtractionStrategy


@dataclass(frozen=True)
class LLMUsageEvent:
    purpose: LLMUsagePurpose
    model: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ClassificationResult:
    doc_type: str
    usage: LLMUsageEvent


@dataclass(frozen=True)
class ResolvedDocumentType:
    doc_type: str
    usage_events: list[LLMUsageEvent] = field(default_factory=list)


@dataclass(frozen=True)
class StrategyExtractionResult:
    raw_output: dict[str, Any]
    input_tokens: int
    output_tokens: int
    strategy: ExtractionStrategy
    source_pages: list[int]
    warnings: list[str] = field(default_factory=list)
    field_source_pages: dict[str, list[int]] | None = None

    @property
    def audit_output(self) -> dict[str, Any]:
        if self.field_source_pages is None:
            return self.raw_output

        return {
            "merged_output": self.raw_output,
            "field_source_pages": self.field_source_pages,
        }

