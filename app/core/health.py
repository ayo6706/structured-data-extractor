from typing import Literal

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession
from sqlalchemy.sql import text

HealthValue = Literal["ok", "error"]
HealthStatus = dict[str, HealthValue]

DATABASE_HEALTH_QUERY = text("SELECT 1")


async def check_database_connection(
    connection: AsyncConnection | AsyncSession,
) -> None:
    await connection.execute(DATABASE_HEALTH_QUERY)


async def check_database_engine(engine: AsyncEngine) -> None:
    async with engine.connect() as connection:
        await check_database_connection(connection)


def healthy_status() -> HealthStatus:
    return {"status": "ok", "database": "ok"}


def unhealthy_status() -> HealthStatus:
    return {"status": "error", "database": "error"}
