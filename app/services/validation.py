from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError

from app.models.extraction import ExtractionStatus
from app.schemas.documents import (
    InvoiceSchema,
    PayslipSchema,
    ReceiptSchema,
    normalize_money,
)
from app.schemas.registry import SchemaRegistry


@dataclass(frozen=True)
class ValidationResult:
    extracted_data: dict[str, Any] | None
    warnings: list[str]
    status: ExtractionStatus


class _ExtractionValidator:
    def validate(self, doc_type: str, raw: dict[str, Any]) -> ValidationResult:
        schema = SchemaRegistry.get_schema(doc_type)

        try:
            model = schema.model_validate(raw)
        except ValidationError as exc:
            return self._recover_partial(schema, raw, exc)

        return ValidationResult(
            extracted_data=model.model_dump(mode="json"),
            warnings=self._quality_warnings(model),
            status=ExtractionStatus.COMPLETED,
        )

    def _recover_partial(
        self,
        schema: type[BaseModel],
        raw: dict[str, Any],
        exc: ValidationError,
    ) -> ValidationResult:
        warnings = self._warnings(schema, raw, exc)
        invalid_fields = self._invalid_fields(exc)
        values: dict[str, Any] = {}
        valid_field_count = 0

        for field_name, field_info in schema.model_fields.items():
            if field_name not in raw or field_name in invalid_fields:
                values[field_name] = self._missing_value(field_info)
                continue

            values[field_name] = self._validated_field_value(
                schema=schema,
                field_name=field_name,
                raw_value=raw[field_name],
            )
            valid_field_count += 1

        status = (
            ExtractionStatus.PARTIAL
            if valid_field_count > 0
            else ExtractionStatus.FAILED
        )
        model = schema.model_construct(**values)
        data = model.model_dump(mode="json")

        return ValidationResult(
            extracted_data=data if valid_field_count > 0 else None,
            warnings=warnings + self._quality_warnings(model),
            status=status,
        )

    def _quality_warnings(self, model: BaseModel) -> list[str]:
        if isinstance(model, InvoiceSchema):
            return self._invoice_warnings(model)
        if isinstance(model, PayslipSchema):
            return self._payslip_warnings(model)
        if isinstance(model, ReceiptSchema):
            return self._receipt_warnings(model)

        return []

    @staticmethod
    def _invoice_warnings(model: InvoiceSchema) -> list[str]:
        if (
            model.subtotal is None
            or model.tax_amount is None
            or model.total_amount is None
        ):
            return []

        expected_total = normalize_money(model.subtotal + model.tax_amount)
        if normalize_money(model.total_amount) == expected_total:
            return []

        return [
            "total_amount: does not equal subtotal plus tax_amount; "
            "document may include discounts, shipping, credits, or rounding"
        ]

    @staticmethod
    def _payslip_warnings(model: PayslipSchema) -> list[str]:
        if (
            model.gross_pay is None
            or model.net_pay is None
            or model.deductions is None
        ):
            return []

        total_deductions = sum(
            (deduction.amount for deduction in model.deductions),
            start=Decimal("0"),
        )
        expected_net_pay = normalize_money(model.gross_pay - total_deductions)
        if normalize_money(model.net_pay) == expected_net_pay:
            return []

        return [
            "net_pay: does not equal gross_pay minus visible deductions; "
            "document may omit deductions or include adjustments"
        ]

    @staticmethod
    def _receipt_warnings(model: ReceiptSchema) -> list[str]:
        if (
            model.subtotal is None
            or model.tax_amount is None
            or model.total_amount is None
        ):
            return []

        expected_total = normalize_money(model.subtotal + model.tax_amount)
        if normalize_money(model.total_amount) == expected_total:
            return []

        return [
            "total_amount: does not equal subtotal plus tax_amount; "
            "receipt may include discounts, fees, credits, or rounding"
        ]

    @staticmethod
    def _invalid_fields(exc: ValidationError) -> set[str]:
        fields = set()
        for error in exc.errors():
            loc = error["loc"]
            if loc and isinstance(loc[0], str):
                fields.add(loc[0])
        return fields

    @classmethod
    def _recovery_input(
        cls,
        schema: type[BaseModel],
        raw: dict[str, Any],
        invalid_fields: set[str],
    ) -> dict[str, Any]:
        values = {}
        for field_name, field_info in schema.model_fields.items():
            if field_name in raw and field_name not in invalid_fields:
                values[field_name] = raw[field_name]
            else:
                values[field_name] = cls._fallback_raw_value(field_info)

        return values

    @classmethod
    def _validated_field_value(
        cls,
        *,
        schema: type[BaseModel],
        field_name: str,
        raw_value: Any,
    ) -> Any:
        field_info = schema.model_fields[field_name]
        candidate = cls._recovery_input(schema, {}, set())
        candidate[field_name] = raw_value

        try:
            model = schema.model_validate(candidate)
        except ValidationError:
            return TypeAdapter(field_info.annotation).validate_python(raw_value)

        return getattr(model, field_name)

    @classmethod
    def _warnings(
        cls,
        schema: type[BaseModel],
        raw: dict[str, Any],
        exc: ValidationError,
    ) -> list[str]:
        warnings = []
        reported_fields = set()

        for error in exc.errors():
            field_name = cls._field_name(error["loc"])
            reported_fields.add(field_name)
            warnings.append(f"{field_name}: {error['msg']}")

        for field_name, field_info in schema.model_fields.items():
            if field_name in raw or field_name in reported_fields:
                continue
            if field_info.is_required():
                warnings.append(f"{field_name}: required field missing")

        return warnings

    @staticmethod
    def _field_name(loc: tuple[Any, ...]) -> str:
        if not loc:
            return "model"

        return str(loc[0])

    @staticmethod
    def _missing_value(field_info: Any) -> Any:
        if field_info.is_required():
            return None

        return field_info.get_default(call_default_factory=True)

    @classmethod
    def _fallback_raw_value(cls, field_info: Any) -> Any:
        if not field_info.is_required():
            return field_info.get_default(call_default_factory=True)

        annotation = cls._unwrap_optional(field_info.annotation)
        origin = get_origin(annotation)

        if annotation is str:
            return ""
        if annotation is int:
            return 0
        if annotation is Decimal:
            return "0"
        if annotation is date:
            return "1970-01-01"
        if origin is list:
            return []
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return {}

        return None

    @classmethod
    def _unwrap_optional(cls, annotation: Any) -> Any:
        origin = get_origin(annotation)
        if origin not in (UnionType, Union):
            return annotation

        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]

        return annotation


def validate_extraction(
    doc_type: str,
    raw: dict[str, Any],
) -> ValidationResult:
    return _ExtractionValidator().validate(doc_type, raw)
