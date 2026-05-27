from fastapi import APIRouter

from app.schemas.registry import SchemaRegistry
from app.schemas.responses import (
    SchemaInfoResponse,
    SchemaListResponse,
)

router = APIRouter(prefix="/schemas", tags=["schemas"])


@router.get("", response_model=SchemaListResponse)
async def list_schemas() -> SchemaListResponse:
    schemas_info = []
    for doc_type in SchemaRegistry.list_types():
        tool_def = SchemaRegistry.get_tool(doc_type)
        schema_def = tool_def["function"]["parameters"]
        schemas_info.append(
            SchemaInfoResponse(name=doc_type, schema_definition=schema_def)
        )

    return SchemaListResponse(schemas=schemas_info)
