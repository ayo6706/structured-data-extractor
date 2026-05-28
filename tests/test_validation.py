from app.models.extraction import ExtractionStatus
from app.services.validation import validate_extraction


def test_validation_returns_complete_for_valid_input() -> None:
    result = validate_extraction(
        "invoice",
        {
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-1",
            "invoice_date": "2026-05-27",
            "subtotal": "1500.00",
            "tax_amount": "150.00",
            "total_amount": "1650.00",
            "currency": "usd",
            "line_items": [
                {"description": "Services", "amount": "1500.00"}
            ],
        },
    )

    assert result.status == ExtractionStatus.COMPLETED
    assert result.warnings == []
    assert result.extracted_data["currency"] == "USD"
    assert result.extracted_data["subtotal"] == "1500.00"


def test_validation_recovers_partial_required_field_missing() -> None:
    result = validate_extraction(
        "invoice",
        {
            "vendor_name": "Acme Corp",
            "invoice_date": "2026-05-27",
            "subtotal": "1500.00",
            "total_amount": "1500.00",
            "currency": "USD",
            "line_items": [
                {"description": "Services", "amount": "1500.00"}
            ],
        },
    )

    assert result.status == ExtractionStatus.PARTIAL
    assert result.extracted_data["invoice_number"] is None
    assert any("invoice_number" in warning for warning in result.warnings)


def test_validation_applies_field_validators_during_partial_recovery() -> None:
    result = validate_extraction(
        "invoice",
        {
            "vendor_name": "Acme Corp",
            "invoice_date": "2026-05-27",
            "subtotal": "1500.00",
            "total_amount": "1500.00",
            "currency": "usd",
            "line_items": [
                {"description": "Services", "amount": "1500.00"}
            ],
        },
    )

    assert result.status == ExtractionStatus.PARTIAL
    assert result.extracted_data["currency"] == "USD"


def test_validation_recovers_partial_invalid_field() -> None:
    result = validate_extraction(
        "receipt",
        {
            "merchant_name": "Corner Shop",
            "receipt_date": "not a date",
            "items": [{"description": "Paper", "amount": "3.50"}],
            "total_amount": "3.50",
            "currency": "USD",
        },
    )

    assert result.status == ExtractionStatus.PARTIAL
    assert result.extracted_data["receipt_date"] is None
    assert any("receipt_date" in warning for warning in result.warnings)


def test_validation_preserves_partial_when_missing_field_affects_invariant():
    result = validate_extraction(
        "invoice",
        {
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-1",
            "invoice_date": "2026-05-27",
            "tax_amount": "150.00",
            "total_amount": "1650.00",
            "currency": "usd",
            "line_items": [
                {"description": "Services", "amount": "1500.00"}
            ],
        },
    )

    assert result.status == ExtractionStatus.PARTIAL
    assert result.extracted_data["subtotal"] is None
    assert result.extracted_data["currency"] == "USD"
    assert result.extracted_data["vendor_name"] == "Acme Corp"


def test_validation_fails_when_no_schema_fields_are_valid() -> None:
    result = validate_extraction("contract", {"foo": "bar"})

    assert result.status == ExtractionStatus.FAILED
    assert result.extracted_data is None
    assert result.warnings


def test_validation_fails_model_level_invariant_errors() -> None:
    result = validate_extraction(
        "invoice",
        {
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-1",
            "invoice_date": "2026-05-27",
            "subtotal": "1500.00",
            "tax_amount": "150.00",
            "total_amount": "1500.00",
            "currency": "USD",
            "line_items": [
                {"description": "Services", "amount": "1500.00"}
            ],
        },
    )

    assert result.status == ExtractionStatus.FAILED
    assert result.extracted_data is None
    assert any("total_amount" in warning for warning in result.warnings)


def test_validation_coerces_decimal_without_warning() -> None:
    result = validate_extraction(
        "receipt",
        {
            "merchant_name": "Corner Shop",
            "receipt_date": "2026-05-27",
            "items": [{"description": "Paper", "amount": "3.50"}],
            "subtotal": "3.00",
            "tax_amount": "0.50",
            "total_amount": "3.50",
            "currency": "USD",
        },
    )

    assert result.status == ExtractionStatus.COMPLETED
    assert result.warnings == []
    assert result.extracted_data["total_amount"] == "3.50"
