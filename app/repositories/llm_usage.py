from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.models.extraction import Extraction
from app.models.llm_usage import LLMUsage, LLMUsagePurpose


class LLMUsageRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(
        self,
        *,
        document_id: UUID,
        extraction_id: UUID | None,
        purpose: LLMUsagePurpose,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> LLMUsage:
        usage = LLMUsage(
            document_id=document_id,
            extraction_id=extraction_id,
            purpose=purpose,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.db.add(usage)
        await self.db.flush()
        return usage

    async def get_doc_type_usage_report_data(
        self,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[Any]:
        stmt = _usage_report_stmt(Extraction.doc_type, from_date, to_date)
        result = await self.db.execute(stmt)
        return list(result.all())

    async def get_strategy_usage_report_data(
        self,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[Any]:
        stmt = _usage_report_stmt(Extraction.strategy, from_date, to_date)
        result = await self.db.execute(stmt)
        return list(result.all())

    async def get_model_usage_report_data(
        self,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[Any]:
        stmt = _usage_report_stmt(LLMUsage.model, from_date, to_date)
        result = await self.db.execute(stmt)
        return list(result.all())

    async def get_purpose_usage_report_data(
        self,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[Any]:
        stmt = _usage_report_stmt(LLMUsage.purpose, from_date, to_date)
        result = await self.db.execute(stmt)
        return list(result.all())


def _usage_report_stmt(
    group_column: Any,
    from_date: datetime | None,
    to_date: datetime | None,
) -> Any:
    stmt = (
        select(
            group_column.label("group_key"),
            LLMUsage.model,
            func.sum(LLMUsage.input_tokens).label("total_input_tokens"),
            func.sum(LLMUsage.output_tokens).label("total_output_tokens"),
        )
        .join(Extraction, LLMUsage.extraction_id == Extraction.id)
        .group_by(group_column, LLMUsage.model)
    )
    if from_date is not None:
        stmt = stmt.where(LLMUsage.created_at >= from_date)
    if to_date is not None:
        stmt = stmt.where(LLMUsage.created_at <= to_date)

    return stmt
