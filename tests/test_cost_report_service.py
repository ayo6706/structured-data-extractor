from decimal import Decimal
from typing import NamedTuple
from unittest.mock import AsyncMock

from app.core.config import ModelTokenPrice
from app.services.cost_report import CostReportService


class CostRow(NamedTuple):
    doc_type: str
    count: int
    avg_duration_ms: float


class UsageCostRow(NamedTuple):
    group_key: str
    model: str
    total_input_tokens: int
    total_output_tokens: int


def _price_for_model(_model: str) -> ModelTokenPrice:
    return ModelTokenPrice(
        input_token_price_usd=Decimal("0.000003"),
        output_token_price_usd=Decimal("0.000015"),
    )


def _unknown_price_for_model(_model: str) -> ModelTokenPrice:
    raise ValueError("unknown model")


def test_build_cost_breakdown_totals_rows() -> None:
    service = CostReportService(
        db=AsyncMock(),
        price_for_model=_price_for_model,
    )
    result = service._build_cost_breakdown(
        stats_rows=[
            CostRow(
                doc_type="invoice",
                count=2,
                avg_duration_ms=100.0,
            )
        ],
        usage_rows=[UsageCostRow("invoice", "test-model", 100, 20)],
        stats_key_attr="doc_type",
        usage_key_attr="group_key",
    )

    assert result.input_tokens == 100
    assert result.output_tokens == 20
    assert result.extractions == 2
    assert result.estimated_cost == Decimal("0.000600")
    assert result.breakdown["invoice"].avg_cost_per_document_usd == Decimal(
        "0.000300"
    )


def test_build_usage_costs_keeps_unknown_models_with_zero_cost() -> None:
    service = CostReportService(
        db=AsyncMock(),
        price_for_model=_unknown_price_for_model,
    )

    result = service._build_usage_costs(
        rows=[UsageCostRow("invoice", "unknown-model", 100, 20)],
        key_attr="group_key",
    )

    assert result["invoice"].total_input_tokens == 100
    assert result["invoice"].total_output_tokens == 20
    assert result["invoice"].estimated_cost_usd == Decimal("0.0")
