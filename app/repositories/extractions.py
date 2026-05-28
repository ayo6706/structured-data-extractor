from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.models.extraction import Extraction, ExtractionStatus


async def create_extraction(
    db: AsyncSession,
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
    db.add(extraction)
    await db.flush()
    await db.refresh(extraction)
    return extraction


async def get_extraction(
    db: AsyncSession,
    extraction_id: UUID,
) -> Extraction | None:
    result = await db.execute(
        select(Extraction).where(Extraction.id == extraction_id)
    )
    return result.scalar_one_or_none()
