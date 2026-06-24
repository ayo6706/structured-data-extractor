from typing import Any
from uuid import UUID

from app.schemas.requests import ExtractionStrategy

REDIS_FALLBACK_WARNING = "Redis unavailable, processed inline"


async def enqueue_extraction(
    arq_pool: Any,
    document_id: UUID,
    doc_type: str | None,
    strategy: ExtractionStrategy | None,
) -> Any:
    return await arq_pool.enqueue_job(
        "extraction_job",
        str(document_id),
        doc_type,
        strategy.value if strategy else None,
    )
