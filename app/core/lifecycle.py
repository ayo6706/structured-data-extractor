import asyncio
import logging

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from fastapi import FastAPI

from app.core.config import (
    LLMAPIKeyConfig,
    arq_settings,
    get_app_settings,
    llm_settings,
)
from app.core.database import get_engine
from app.core.health import check_database_engine

logger = logging.getLogger(__name__)
DATABASE_STARTUP_TIMEOUT_SECONDS = 5
_MODEL_PREFIX_TO_ENV_VAR: dict[str, str] = {
    "gemini/": "GEMINI_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "openai/": "OPENAI_API_KEY",
    "gpt-": "OPENAI_API_KEY",
}


def get_arq_redis_settings() -> RedisSettings:
    url = arq_settings.REDIS_URL
    database = int(url.path.lstrip("/")) if url.path and url.path != "/" else 0
    return RedisSettings(
        host=url.host or "localhost",
        port=url.port or 6379,
        database=database,
        password=url.password,
    )


def validate_llm_api_keys(
    api_key_settings: LLMAPIKeyConfig | None = None,
) -> None:
    keys = api_key_settings or LLMAPIKeyConfig()
    models = {llm_settings.EXTRACTION_MODEL, llm_settings.CLASSIFIER_MODEL}
    for model in models:
        for prefix, env_var in _MODEL_PREFIX_TO_ENV_VAR.items():
            if model.startswith(prefix) and not getattr(keys, env_var):
                raise RuntimeError(
                    f"Model '{model}' requires environment variable "
                    f"'{env_var}' to be set."
                )


async def startup(app: FastAPI | None = None) -> None:
    validate_llm_api_keys()

    settings = get_app_settings()
    try:
        settings.STORAGE_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Local storage directory initialized at %s",
            settings.STORAGE_LOCAL_DIR,
        )
    except Exception as exc:
        logger.error("Failed to create local storage directory: %s", exc)
        raise

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

    if app is not None:
        app.state.arq_pool = await _create_arq_pool()


async def shutdown(app: FastAPI | None = None) -> None:
    if app is not None:
        await _close_arq_pool(getattr(app.state, "arq_pool", None))
        app.state.arq_pool = None

    engine = get_engine()
    await engine.dispose()
    clear_engine_cache = getattr(get_engine, "cache_clear", None)
    if clear_engine_cache is not None:
        clear_engine_cache()
    logger.info("Database connection closed.")


async def _create_arq_pool() -> ArqRedis | None:
    try:
        pool = await create_pool(get_arq_redis_settings())
        logger.info("Successfully connected to Redis for Arq.")
        return pool
    except Exception as exc:
        logger.warning(
            "Redis unavailable for Arq; async extraction will fall back "
            "to inline processing: %s",
            exc,
        )
        return None


async def _close_arq_pool(pool: ArqRedis | None) -> None:
    if pool is None:
        return

    await pool.close()
    logger.info("Redis connection closed.")
