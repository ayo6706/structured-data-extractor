import asyncio
import logging

from app.core.database import get_engine
from app.core.health import check_database_engine

logger = logging.getLogger(__name__)
DATABASE_STARTUP_TIMEOUT_SECONDS = 5


async def startup() -> None:
    try:
        engine = get_engine()
        await asyncio.wait_for(
            check_database_engine(engine),
            timeout=DATABASE_STARTUP_TIMEOUT_SECONDS,
        )
        logger.info("Successfully connected to the database.")
    except TimeoutError:
        logger.error(
            "Timed out connecting to the database after %s seconds.",
            DATABASE_STARTUP_TIMEOUT_SECONDS,
        )
        raise
    except Exception as exc:
        logger.error("Failed to connect to the database: %s", exc)
        raise


async def shutdown() -> None:
    engine = get_engine()
    await engine.dispose()
    clear_engine_cache = getattr(get_engine, "cache_clear", None)
    if clear_engine_cache is not None:
        clear_engine_cache()
    logger.info("Database connection closed.")
