from fastapi import APIRouter

from app.api.v1.endpoints import (
    documents,
    health,
    schemas_endpoint,
)

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(schemas_endpoint.router)
api_router.include_router(documents.router)
