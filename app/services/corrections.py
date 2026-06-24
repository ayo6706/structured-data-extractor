from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    CorrectionValidationError,
)
from app.models.correction import ExtractionCorrection
from app.models.extraction import Extraction
from app.repositories.corrections import CorrectionRepository
from app.repositories.transactions import commit, refresh
from app.schemas.requests import CorrectionRequest
from app.schemas.responses import CorrectionResponse
from app.services.validation import validate_extraction


class CorrectionService:
    def __init__(self, db: AsyncSession | None = None) -> None:
        self.db = db

    async def correct(
        self,
        extraction: Extraction,
        request: CorrectionRequest,
    ) -> CorrectionResponse:
        if self.db is None:
            raise RuntimeError("CorrectionService requires a database session")

        correction = await self.create(
            self.db,
            extraction=extraction,
            request=request,
        )
        return CorrectionResponse(
            correction_id=correction.id,
            extraction_id=correction.extraction_id,
            original_data=correction.original_data,
            corrected_data=correction.corrected_data,
            correction_diff=correction.correction_diff,
            warnings=correction.warnings or [],
            corrected_by=correction.corrected_by,
            note=correction.note,
            created_at=correction.created_at,
        )

    async def create(
        self,
        db: AsyncSession,
        *,
        extraction: Extraction,
        request: CorrectionRequest,
    ) -> ExtractionCorrection:
        validation = validate_extraction(
            extraction.doc_type,
            request.corrected_data,
        )
        if validation.extracted_data is None:
            raise CorrectionValidationError(validation.warnings)

        corrected_data = validation.extracted_data
        original_data = extraction.extracted_data or {}
        correction_diff = diff_extraction_data(original_data, corrected_data)

        correction = await CorrectionRepository(db).create(
            extraction_id=extraction.id,
            original_data=extraction.extracted_data,
            corrected_data=corrected_data,
            correction_diff=correction_diff,
            warnings=validation.warnings,
            corrected_by=request.corrected_by,
            note=request.note,
        )
        await commit(db)
        await refresh(db, correction)
        return correction


def diff_extraction_data(
    original: dict[str, Any],
    corrected: dict[str, Any],
) -> dict[str, Any]:
    added = {}
    removed = {}
    changed = {}

    for field_name in sorted(original.keys() | corrected.keys()):
        original_value = original.get(field_name)
        corrected_value = corrected.get(field_name)

        if field_name not in original:
            added[field_name] = corrected_value
            continue

        if field_name not in corrected:
            removed[field_name] = original_value
            continue

        if original_value != corrected_value:
            changed[field_name] = {
                "from": original_value,
                "to": corrected_value,
            }

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
    }
