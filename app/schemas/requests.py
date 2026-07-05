from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ExtractionStrategy(StrEnum):
    FULL = "full"
    PAGE_BY_PAGE = "page_by_page"
    SMART = "smart"

class CorrectionRequest(BaseModel):
    corrected_data: dict[str, Any] = Field(
        description="Human-corrected extraction fields"
    )
    corrected_by: str | None = Field(
        None,
        max_length=255,
        description="Optional reviewer identifier",
    )
    note: str | None = Field(
        None,
        max_length=1000,
        description="Optional correction note",
    )
