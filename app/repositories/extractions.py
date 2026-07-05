from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.models.extraction import Extraction, ExtractionStatus


class ExtractionRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, extraction: Extraction) -> Extraction:
        self.db.add(extraction)
        await self.db.flush()
        return extraction

    async def create_from_fields(
        self,
        *,
        document_id: UUID,
        doc_type: str,
        status: ExtractionStatus,
        extracted_data: dict[str, Any] | None,
        confidence_map: dict[str, float],
        warnings: list[str],
        raw_tool_output: dict[str, Any] | None,
        model_used: str,
        input_tokens: int,
        output_tokens: int,
        extraction_duration_ms: int,
        source_pages: list[int] | None,
        strategy: str,
    ) -> Extraction:
        extraction = Extraction(
            document_id=document_id,
            doc_type=doc_type,
            status=status,
            extracted_data=extracted_data,
            confidence_map=confidence_map,
            warnings=warnings,
            raw_tool_output=raw_tool_output,
            strategy=strategy,
            source_pages=source_pages,
            model_used=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            extraction_duration_ms=extraction_duration_ms,
        )
        self.db.add(extraction)
        await self.db.flush()
        await self.db.refresh(extraction)
        return extraction

    async def get_by_id(self, extraction_id: UUID) -> Extraction | None:
        result = await self.db.execute(
            select(Extraction).where(Extraction.id == extraction_id)
        )
        return result.scalar_one_or_none()

    async def get_cost_report_data(
        self,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[Any]:
        stmt = select(
            Extraction.doc_type,
            func.count(Extraction.id).label("count"),
            func.avg(Extraction.extraction_duration_ms).label(
                "avg_duration_ms"
            ),
        ).group_by(Extraction.doc_type)
        if from_date is not None:
            stmt = stmt.where(Extraction.created_at >= from_date)
        if to_date is not None:
            stmt = stmt.where(Extraction.created_at <= to_date)

        result = await self.db.execute(stmt)
        return list(result.all())

    async def get_strategy_cost_report_data(
        self,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[Any]:
        stmt = select(
            Extraction.strategy,
            func.count(Extraction.id).label("count"),
            func.avg(Extraction.extraction_duration_ms).label(
                "avg_duration_ms"
            ),
        ).group_by(Extraction.strategy)
        if from_date is not None:
            stmt = stmt.where(Extraction.created_at >= from_date)
        if to_date is not None:
            stmt = stmt.where(Extraction.created_at <= to_date)

        result = await self.db.execute(stmt)
        return list(result.all())
