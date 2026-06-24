from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.correction import ExtractionCorrection


class CorrectionRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(
        self,
        *,
        extraction_id: UUID,
        original_data: dict[str, Any] | None,
        corrected_data: dict[str, Any],
        correction_diff: dict[str, Any],
        warnings: list[str],
        corrected_by: str | None,
        note: str | None,
    ) -> ExtractionCorrection:
        correction = ExtractionCorrection(
            extraction_id=extraction_id,
            original_data=original_data,
            corrected_data=corrected_data,
            correction_diff=correction_diff,
            warnings=warnings,
            corrected_by=corrected_by,
            note=note,
        )
        self.db.add(correction)
        await self.db.flush()
        await self.db.refresh(correction)
        return correction
