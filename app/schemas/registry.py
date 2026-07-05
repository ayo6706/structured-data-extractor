from pydantic import BaseModel

from app.schemas.documents import (
    ContractSchema,
    InvoiceSchema,
    PayslipSchema,
    ReceiptSchema,
)


class SchemaRegistry:
    _registry: dict[str, type[BaseModel]] = {
        "invoice": InvoiceSchema,
        "contract": ContractSchema,
        "payslip": PayslipSchema,
        "receipt": ReceiptSchema,
    }

    @classmethod
    def get_tool(cls, doc_type: str) -> dict:
        schema = cls.get_schema(doc_type)
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
    def get_schema(cls, doc_type: str) -> type[BaseModel]:
        if doc_type not in cls._registry:
            raise KeyError(f"Unknown document type: {doc_type}")

        return cls._registry[doc_type]

    @classmethod
    def list_types(cls) -> list[str]:
        return list(cls._registry.keys())
