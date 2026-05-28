from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_app_settings
from app.core.database import get_db
from app.infrastructure.storage import LocalStorage, StorageBackend
from app.integrations.llm.litellm_client import LiteLLMClient
from app.services.classifier import ClassifierService
from app.services.extraction import ExtractionService

DbDep = Annotated[AsyncSession, Depends(get_db)]


@lru_cache
def get_llm_client() -> LiteLLMClient:
    return LiteLLMClient()


LLMClientDep = Annotated[LiteLLMClient, Depends(get_llm_client)]


def get_classifier_service(llm_client: LLMClientDep) -> ClassifierService:
    return ClassifierService(llm_client=llm_client)


ClassifierDep = Annotated[ClassifierService, Depends(get_classifier_service)]


@lru_cache
def get_storage() -> StorageBackend:
    settings = get_app_settings()
    if settings.STORAGE_BACKEND == "local":
        return LocalStorage(base_dir=settings.STORAGE_LOCAL_DIR)
    raise NotImplementedError(
        f"Storage backend '{settings.STORAGE_BACKEND}' is not implemented."
    )


StorageDep = Annotated[StorageBackend, Depends(get_storage)]


def get_extraction_service(
    classifier: ClassifierDep,
    llm_client: LLMClientDep,
    storage: StorageDep,
) -> ExtractionService:
    return ExtractionService(
        classifier=classifier,
        llm_client=llm_client,
        storage=storage,
    )


ExtractionDep = Annotated[ExtractionService, Depends(get_extraction_service)]
