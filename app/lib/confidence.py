from decimal import Decimal, InvalidOperation
from typing import Any

from app.schemas.registry import SchemaRegistry

PLACEHOLDER_VALUES = {"n/a", "unknown", "not stated"}
MONEY_FIELD_MARKERS = ("amount", "pay", "price", "subtotal", "total")


def score_confidence_map(
    *,
    doc_type: str,
    extracted_data: dict[str, Any] | None,
) -> dict[str, float]:
    schema = SchemaRegistry.get_schema(doc_type)
    data = extracted_data or {}

    return {
        field_name: _score_field(
            field_name=field_name,
            value=data.get(field_name),
            required=field_info.is_required(),
        )
        for field_name, field_info in schema.model_fields.items()
    }


def _score_field(
    *,
    field_name: str,
    value: Any,
    required: bool,
) -> float:
    if value is None:
        return 0.0

    if _is_placeholder(value):
        return 0.2

    score = 0.85 if required else 0.70
    if _is_suspicious_amount(field_name, value):
        return min(score, 0.60)

    return score


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in (
        PLACEHOLDER_VALUES
    )


def _is_suspicious_amount(field_name: str, value: Any) -> bool:
    if not any(marker in field_name for marker in MONEY_FIELD_MARKERS):
        return False

    if isinstance(value, int | float | Decimal):
        return value < 0

    if isinstance(value, str):
        try:
            return Decimal(value) < 0
        except InvalidOperation:
            return False

    return False
