from app.lib.confidence import score_confidence_map


def test_confidence_scores_required_and_optional_valid_fields() -> None:
    scores = score_confidence_map(
        doc_type="invoice",
        extracted_data={
            "vendor_name": "Acme Corp",
            "vendor_address": "1 Main Street",
            "invoice_number": "INV-1",
            "invoice_date": "2026-05-27",
            "due_date": None,
            "subtotal": "100.00",
            "tax_amount": "10.00",
            "total_amount": "110.00",
            "currency": "USD",
            "line_items": [{"description": "Services", "amount": "100.00"}],
            "payment_terms": "Net 30",
        },
    )

    assert scores["vendor_name"] == 0.85
    assert scores["vendor_address"] == 0.70
    assert scores["due_date"] == 0.0
    assert set(scores) == {
        "vendor_name",
        "vendor_address",
        "invoice_number",
        "invoice_date",
        "due_date",
        "subtotal",
        "tax_amount",
        "total_amount",
        "currency",
        "line_items",
        "payment_terms",
    }


def test_confidence_scores_placeholders() -> None:
    scores = score_confidence_map(
        doc_type="receipt",
        extracted_data={
            "merchant_name": "unknown",
            "receipt_date": "2026-05-27",
            "items": [{"description": "Paper", "amount": "3.50"}],
            "total_amount": "3.50",
            "currency": "USD",
        },
    )

    assert scores["merchant_name"] == 0.2


def test_confidence_caps_negative_amounts() -> None:
    scores = score_confidence_map(
        doc_type="invoice",
        extracted_data={
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-1",
            "invoice_date": "2026-05-27",
            "subtotal": "-100.00",
            "total_amount": "-100.00",
            "currency": "USD",
            "line_items": [{"description": "Refund", "amount": "-100.00"}],
        },
    )

    assert scores["subtotal"] == 0.60
    assert scores["total_amount"] == 0.60
