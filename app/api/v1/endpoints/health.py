from fastapi import APIRouter, HTTPException

from app.api.dependencies import DbDep
from app.core.health import (
    check_database_connection,
    healthy_status,
    unhealthy_status,
)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check(db: DbDep) -> dict[str, str]:
    try:
        await check_database_connection(db)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail=unhealthy_status(),
        ) from None

    return healthy_status()
