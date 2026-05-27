from pydantic import BaseModel

from app.schemas.documents import (
    ContractSchema,
    InvoiceSchema,
    PayslipSchema,
    ReceiptSchema,
)


class SchemaRegistry:
    """Registry for document schemas and LLM tool definitions."""

    _registry: dict[str, type[BaseModel]] = {
        "invoice": InvoiceSchema,
        "contract": ContractSchema,
        "payslip": PayslipSchema,
        "receipt": ReceiptSchema,
    }

    @classmethod
    def get_tool(cls, doc_type: str) -> dict:
        """Returns OpenAI-format tool definition."""
        if doc_type not in cls._registry:
            raise KeyError(f"Unknown document type: {doc_type}")

        schema = cls._registry[doc_type]
        return {
            "type": "function",
            "function": {
                "name": f"extract_{doc_type}",
                "description": (
                    f"Extract structured {doc_type} data from document text"
                ),
                "parameters": schema.model_json_schema(),
            },
        }

    @classmethod
    def validate(cls, doc_type: str, raw: dict) -> BaseModel:
        """Validates raw dict against the schema for doc_type."""
        if doc_type not in cls._registry:
            raise KeyError(f"Unknown document type: {doc_type}")

        schema = cls._registry[doc_type]
        return schema.model_validate(raw)

    @classmethod
    def list_types(cls) -> list[str]:
        """Returns all registered document type names."""
        return list(cls._registry.keys())
