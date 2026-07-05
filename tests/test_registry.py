import pytest
from pydantic import BaseModel, Field

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


@pytest.mark.parametrize("doc_type", SchemaRegistry.list_types())
def test_registry_get_tool_for_each_document_type(doc_type: str) -> None:
    tool = SchemaRegistry.get_tool(doc_type)

    assert tool["type"] == "function"
    assert tool["function"]["name"] == f"extract_{doc_type}"
    assert tool["function"]["description"]
    assert tool["function"]["parameters"]["type"] == "object"
    assert tool["function"]["parameters"]["properties"]


def test_registry_get_tool_unknown():
    with pytest.raises(KeyError) as exc_info:
        SchemaRegistry.get_tool("unknown_doc")

    assert "Unknown document type: unknown_doc" in str(exc_info.value)


def test_registry_picks_up_new_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BankStatementSchema(BaseModel):
        account_holder: str = Field(description="Name of account holder")

    registry = {
        **SchemaRegistry._registry,
        "bank_statement": BankStatementSchema,
    }
    monkeypatch.setattr(SchemaRegistry, "_registry", registry)

    assert "bank_statement" in SchemaRegistry.list_types()
    tool = SchemaRegistry.get_tool("bank_statement")
    assert tool["function"]["name"] == "extract_bank_statement"
    assert "account_holder" in tool["function"]["parameters"]["properties"]
