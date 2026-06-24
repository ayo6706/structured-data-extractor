from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import ModelTokenPrice
from app.repositories.extractions import ExtractionRepository
from app.repositories.llm_usage import LLMUsageRepository
from app.schemas.responses import CostReportResponse, DocTypeCost, UsageCost


class PriceResolver(Protocol):
    def __call__(self, model: str) -> ModelTokenPrice:
        pass


@dataclass(frozen=True)
class _CostBreakdown:
    breakdown: dict[str, DocTypeCost]
    input_tokens: int
    output_tokens: int
    estimated_cost: Decimal
    extractions: int


class CostReportService:
    def __init__(
        self,
        *,
        db: AsyncSession,
        price_for_model: PriceResolver,
    ) -> None:
        self.db = db
        self.price_for_model = price_for_model

    async def generate(
        self,
        from_date: Any = None,
        to_date: Any = None,
    ) -> CostReportResponse:
        extractions = ExtractionRepository(self.db)
        usage = LLMUsageRepository(self.db)

        doc_type_rows = await extractions.get_cost_report_data(
            from_date, to_date
        )
        strategy_rows = await extractions.get_strategy_cost_report_data(
            from_date, to_date
        )
        doc_type_usage_rows = await usage.get_doc_type_usage_report_data(
            from_date, to_date
        )
        strategy_usage_rows = await usage.get_strategy_usage_report_data(
            from_date, to_date
        )
        model_usage_rows = await usage.get_model_usage_report_data(
            from_date, to_date
        )
        purpose_usage_rows = await usage.get_purpose_usage_report_data(
            from_date, to_date
        )

        doc_type_costs = self._build_cost_breakdown(
            stats_rows=doc_type_rows,
            usage_rows=doc_type_usage_rows,
            stats_key_attr="doc_type",
            usage_key_attr="group_key",
        )
        strategy_costs = self._build_cost_breakdown(
            stats_rows=strategy_rows,
            usage_rows=strategy_usage_rows,
            stats_key_attr="strategy",
            usage_key_attr="group_key",
        )
        return CostReportResponse(
            by_doc_type=doc_type_costs.breakdown,
            by_strategy=strategy_costs.breakdown,
            by_model=self._build_usage_costs(
                rows=model_usage_rows,
                key_attr="group_key",
            ),
            by_purpose=self._build_usage_costs(
                rows=purpose_usage_rows,
                key_attr="group_key",
            ),
            total_input_tokens=doc_type_costs.input_tokens,
            total_output_tokens=doc_type_costs.output_tokens,
            total_estimated_cost_usd=doc_type_costs.estimated_cost,
            total_extractions=doc_type_costs.extractions,
        )

    def _build_cost_breakdown(
        self,
        *,
        stats_rows: list[Any],
        usage_rows: list[Any],
        stats_key_attr: str,
        usage_key_attr: str,
    ) -> _CostBreakdown:
        breakdown = {
            str(getattr(row, stats_key_attr)): self._empty_cost_row(row)
            for row in stats_rows
        }
        total_extractions = sum(item.count for item in breakdown.values())

        usage_totals = self._build_usage_costs(
            rows=usage_rows,
            key_attr=usage_key_attr,
        )
        for key, usage in usage_totals.items():
            current = breakdown.get(key)
            count = current.count if current else 0
            avg_extraction_ms = current.avg_extraction_ms if current else 0.0
            avg_cost = (
                usage.estimated_cost_usd / Decimal(count)
                if count > 0
                else Decimal("0.0")
            )
            breakdown[key] = DocTypeCost(
                count=count,
                avg_extraction_ms=avg_extraction_ms,
                total_input_tokens=usage.total_input_tokens,
                total_output_tokens=usage.total_output_tokens,
                estimated_cost_usd=usage.estimated_cost_usd,
                avg_cost_per_document_usd=avg_cost,
            )

        total_input_tokens = sum(
            item.total_input_tokens for item in usage_totals.values()
        )
        total_output_tokens = sum(
            item.total_output_tokens for item in usage_totals.values()
        )
        total_estimated_cost_usd = sum(
            (item.estimated_cost_usd for item in usage_totals.values()),
            Decimal("0.0"),
        )

        return _CostBreakdown(
            breakdown=breakdown,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            estimated_cost=total_estimated_cost_usd,
            extractions=total_extractions,
        )

    def _build_usage_costs(
        self,
        *,
        rows: list[Any],
        key_attr: str,
    ) -> dict[str, UsageCost]:
        costs: dict[str, UsageCost] = {}

        for row in rows:
            key = str(getattr(row, key_attr))
            price = self.price_for_model(str(row.model))
            input_tokens = (
                int(row.total_input_tokens)
                if row.total_input_tokens is not None
                else 0
            )
            output_tokens = (
                int(row.total_output_tokens)
                if row.total_output_tokens is not None
                else 0
            )
            estimated_cost = (input_tokens * price.input_token_price_usd) + (
                output_tokens * price.output_token_price_usd
            )
            current = costs.get(key)
            if current is None:
                costs[key] = UsageCost(
                    total_input_tokens=input_tokens,
                    total_output_tokens=output_tokens,
                    estimated_cost_usd=estimated_cost,
                )
                continue

            costs[key] = UsageCost(
                total_input_tokens=current.total_input_tokens + input_tokens,
                total_output_tokens=current.total_output_tokens
                + output_tokens,
                estimated_cost_usd=current.estimated_cost_usd
                + estimated_cost,
            )

        return costs

    def _empty_cost_row(self, row: Any) -> DocTypeCost:
        count = int(row.count) if row.count is not None else 0
        avg_extraction_ms = (
            float(row.avg_duration_ms)
            if row.avg_duration_ms is not None
            else 0.0
        )
        return DocTypeCost(
            count=count,
            avg_extraction_ms=avg_extraction_ms,
            total_input_tokens=0,
            total_output_tokens=0,
            estimated_cost_usd=Decimal("0.0"),
            avg_cost_per_document_usd=Decimal("0.0"),
        )
