from decimal import Decimal
from pathlib import Path

import pytest

from app.schemas.requests import ExtractionStrategy
from app.services.types import StrategyExtractionResult
from scripts.evaluate_strategies import (
    StrategyEvaluationResult,
    evaluate_strategies,
    render_strategy_evaluation_markdown,
)


class FakePageRunner:
    async def extract_from_pages(
        self,
        *,
        pages: list[str],
        doc_type: str,
        strategy: ExtractionStrategy,
    ) -> StrategyExtractionResult:
        return StrategyExtractionResult(
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
            strategy=strategy,
            source_pages=[0],
        )


@pytest.mark.asyncio
async def test_evaluate_strategies_summarizes_each_strategy() -> None:
    results = await evaluate_strategies(
        page_runner=FakePageRunner(),
        pages=["invoice text"],
        doc_type="invoice",
        strategies=(ExtractionStrategy.FULL, ExtractionStrategy.SMART),
    )

    assert [result.strategy for result in results] == [
        ExtractionStrategy.FULL,
        ExtractionStrategy.SMART,
    ]
    assert all(result.status == "completed" for result in results)
    assert all(result.input_tokens == 100 for result in results)
    assert all("vendor_name" in result.extracted_fields for result in results)


def test_render_strategy_evaluation_markdown() -> None:
    evaluation = render_strategy_evaluation_markdown(
        pdf_path=Path("sample.pdf"),
        doc_type="invoice",
        results=[
            StrategyEvaluationResult(
                strategy=ExtractionStrategy.FULL,
                status="completed",
                source_pages=[0],
                input_tokens=1,
                output_tokens=1,
                estimated_cost_usd=Decimal("0.000018"),
                extracted_fields=["vendor_name"],
                warnings=[],
            )
        ],
    )

    assert "# Strategy Comparison" in evaluation
    assert "| full | completed | 0 | 1 | 1 | 0.000018 | vendor_name | - |" in (
        evaluation
    )
