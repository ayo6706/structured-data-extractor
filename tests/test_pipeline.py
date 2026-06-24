from unittest.mock import AsyncMock

import pytest

from app.schemas.requests import ExtractionStrategy
from app.services.extraction_pipeline import QualityPipeline
from app.services.types import StrategyExtractionResult


class FakePageRunner:
    def __init__(self) -> None:
        self.extract_from_pages = AsyncMock(
            return_value=StrategyExtractionResult(
                raw_output={
                    "vendor_name": "Acme Corp",
                    "invoice_number": "INV-1",
                    "invoice_date": "2026-05-27",
                    "subtotal": "100.00",
                    "tax_amount": "10.00",
                    "total_amount": "110.00",
                    "currency": "USD",
                    "line_items": [],
                },
                input_tokens=100,
                output_tokens=20,
                strategy=ExtractionStrategy.FULL,
                source_pages=[0],
            )
        )


@pytest.mark.asyncio
async def test_pipeline_runs_with_resolved_type_and_strategy() -> None:
    page_runner = FakePageRunner()
    result = await QualityPipeline(page_runner).run_resolved(
        pages=["invoice text"],
        doc_type="invoice",
        strategy=ExtractionStrategy.SMART,
    )

    assert result.doc_type == "invoice"
    assert result.validation.status.value == "completed"
    assert result.extraction.input_tokens == 100
    assert result.confidence_map["vendor_name"] == 0.85
    assert result.warnings == []
    page_runner.extract_from_pages.assert_awaited_once_with(
        pages=["invoice text"],
        doc_type="invoice",
        strategy=ExtractionStrategy.SMART,
    )
