from functools import lru_cache
from typing import Annotated

from arq.connections import ArqRedis
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import cost_settings, get_app_settings
from app.core.database import get_db, get_session_factory
from app.factories.document_processor import DocumentProcessorFactory
from app.infrastructure.storage import LocalStorage, StorageBackend
from app.integrations.llm.litellm_client import LiteLLMClient
from app.services.classifier import ClassifierService
from app.services.document_processor import DocumentProcessor
from app.services.documents import DocumentService
from app.services.page_extraction import PageStrategyRunner
from app.services.tool_call_extractor import ToolCallExtractor

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


def get_document_processor_factory(
    classifier: ClassifierDep,
    llm_client: LLMClientDep,
    storage: StorageDep,
) -> DocumentProcessorFactory:
    tool_extractor = ToolCallExtractor(llm_client=llm_client)
    page_runner = PageStrategyRunner(text_extractor=tool_extractor)
    return DocumentProcessorFactory(
        classifier=classifier,
        page_runner=page_runner,
        storage=storage,
        model=tool_extractor.model,
    )


DocumentProcessorFactoryDep = Annotated[
    DocumentProcessorFactory,
    Depends(get_document_processor_factory),
]


def get_extraction_service(
    db: DbDep,
    processor_factory: DocumentProcessorFactoryDep,
) -> DocumentProcessor:
    return processor_factory.create(db)


ExtractionDep = Annotated[DocumentProcessor, Depends(get_extraction_service)]


def get_arq_pool(request: Request) -> ArqRedis | None:
    return getattr(request.app.state, "arq_pool", None)


ArqPoolDep = Annotated[ArqRedis | None, Depends(get_arq_pool)]


def get_document_service(
    db: DbDep,
    processor: ExtractionDep,
    processor_factory: DocumentProcessorFactoryDep,
    arq_pool: ArqPoolDep,
) -> DocumentService:
    return DocumentService(
        db=db,
        processor=processor,
        processor_factory=processor_factory,
        arq_pool=arq_pool,
        session_factory=get_session_factory(),
        price_for_model=cost_settings.price_for_model,
    )


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
