from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.documents import (
    ContractSchema,
    InvoiceSchema,
    PayslipSchema,
    ReceiptSchema,
)
from app.schemas.requests import ExtractRequest


@pytest.mark.parametrize(
    ("schema_class", "field_name", "nested_definition", "nested_field"),
    [
        (InvoiceSchema, "vendor_name", "LineItem", "amount"),
        (ContractSchema, "parties", "Party", "name"),
        (PayslipSchema, "employee_name", "Deduction", "amount"),
        (ReceiptSchema, "merchant_name", "LineItem", "amount"),
    ],
)
def test_document_schemas_include_field_descriptions(
    schema_class, field_name, nested_definition, nested_field
):
    schema = schema_class.model_json_schema()

    assert schema["type"] == "object"
    assert "description" in schema["properties"][field_name]
    assert (
        "description"
        in schema["$defs"][nested_definition]["properties"][nested_field]
    )


def test_invoice_schema_validation():
    """Verify that valid dictionaries validate correctly."""
    valid_data = {
        "vendor_name": "Test Vendor",
        "invoice_number": "INV-001",
        "invoice_date": "2026-05-26",
        "subtotal": "100.00",
        "total_amount": "120.00",
        "currency": "USD",
        "line_items": [
            {
                "description": "Consulting",
                "quantity": 1,
                "unit_price": "100.00",
                "amount": "100.00",
            }
        ],
    }

    model = InvoiceSchema.model_validate(valid_data)
    assert model.vendor_name == "Test Vendor"
    assert model.subtotal == Decimal("100.00")
    assert model.total_amount == Decimal("120.00")
    assert model.line_items[0].amount == Decimal("100.00")


def test_invoice_schema_invalid_data():
    invalid_data = {"vendor_name": "Test Vendor"}

    with pytest.raises(ValidationError) as exc_info:
        InvoiceSchema.model_validate(invalid_data)

    errors = exc_info.value.errors()
    error_fields = [e["loc"][0] for e in errors]
    assert "invoice_number" in error_fields
    assert "invoice_date" in error_fields


def test_invoice_schema_rejects_inconsistent_dates_and_totals():
    valid_data = {
        "vendor_name": "Test Vendor",
        "invoice_number": "INV-001",
        "invoice_date": "2026-05-26",
        "due_date": "2026-05-25",
        "subtotal": "100.00",
        "tax_amount": "20.00",
        "total_amount": "119.00",
        "currency": "USD",
        "line_items": [],
    }

    with pytest.raises(ValidationError):
        InvoiceSchema.model_validate(valid_data)


def test_contract_schema_rejects_invalid_termination_date():
    with pytest.raises(ValidationError):
        ContractSchema.model_validate(
            {
                "parties": [{"name": "A"}],
                "effective_date": "2026-05-26",
                "termination_date": "2026-05-25",
                "key_obligations": [],
            }
        )


def test_payslip_schema_rejects_invalid_net_pay():
    with pytest.raises(ValidationError):
        PayslipSchema.model_validate(
            {
                "employee_name": "Ada",
                "employer_name": "Acme",
                "pay_period_start": "2026-05-01",
                "pay_period_end": "2026-05-31",
                "gross_pay": "1000.00",
                "net_pay": "900.00",
                "currency": "USD",
                "deductions": [{"name": "Tax", "amount": "50.00"}],
                "pay_date": "2026-05-31",
            }
        )


def test_receipt_schema_rejects_inconsistent_total():
    with pytest.raises(ValidationError):
        ReceiptSchema.model_validate(
            {
                "merchant_name": "Store",
                "receipt_date": "2026-05-26",
                "items": [],
                "subtotal": "10.00",
                "tax_amount": "1.00",
                "total_amount": "10.50",
                "currency": "USD",
            }
        )


def test_extract_request_restricts_strategy_and_doc_type():
    assert ExtractRequest.model_validate(
        {"doc_type": "invoice", "strategy": "smart"}
    )

    with pytest.raises(ValidationError):
        ExtractRequest.model_validate({"doc_type": "unknown"})

    with pytest.raises(ValidationError):
        ExtractRequest.model_validate({"strategy": "custom"})
