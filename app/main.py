import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.exception_handlers import register_exception_handlers
from app.api.v1.router import api_router
from app.core import lifecycle
from app.core.config import get_app_settings

logger = logging.getLogger(__name__)
settings = get_app_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("%s starting up.", settings.PROJECT_NAME)

    await lifecycle.startup(app)

    try:
        yield
    finally:
        logger.info("%s shutting down.", settings.PROJECT_NAME)
        await lifecycle.shutdown(app)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for extracting typed structured data from PDFs via LLMs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(api_router, prefix=settings.API_V1_PREFIX)
