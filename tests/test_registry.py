import pytest

from app.schemas.documents import InvoiceSchema
from app.schemas.registry import SchemaRegistry


def test_registry_list_types():
    assert SchemaRegistry.list_types() == [
        "invoice",
        "contract",
        "payslip",
        "receipt",
    ]


def test_registry_get_tool():
    tool = SchemaRegistry.get_tool("invoice")

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "extract_invoice"
    assert "description" in tool["function"]
    params = tool["function"]["parameters"]
    assert params["type"] == "object"
    assert "vendor_name" in params["properties"]


def test_registry_get_tool_unknown():
    with pytest.raises(KeyError) as exc_info:
        SchemaRegistry.get_tool("unknown_doc")

    assert "Unknown document type: unknown_doc" in str(exc_info.value)


def test_registry_validate():
    raw_data = {
        "vendor_name": "Test",
        "invoice_number": "123",
        "invoice_date": "2026-05-26",
        "subtotal": 10,
        "total_amount": 10,
        "currency": "USD",
        "line_items": [],
    }

    model = SchemaRegistry.validate("invoice", raw_data)
    assert isinstance(model, InvoiceSchema)
    assert model.vendor_name == "Test"
