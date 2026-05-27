from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.integrations.llm.litellm_client import LiteLLMClient
from app.services.classifier import ClassifierService
from app.services.extraction import ExtractService

DbDep = Annotated[AsyncSession, Depends(get_db)]


@lru_cache
def get_llm_client() -> LiteLLMClient:
    return LiteLLMClient()


LLMClientDep = Annotated[LiteLLMClient, Depends(get_llm_client)]


def get_classifier_service(llm_client: LLMClientDep) -> ClassifierService:
    return ClassifierService(llm_client=llm_client)


ClassifierDep = Annotated[ClassifierService, Depends(get_classifier_service)]


def get_extract_service(classifier: ClassifierDep) -> ExtractService:
    return ExtractService(classifier=classifier)


ExtractDep = Annotated[ExtractService, Depends(get_extract_service)]
