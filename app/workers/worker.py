import logging
from uuid import UUID

from app.api.dependencies import (
    get_classifier_service,
    get_llm_client,
    get_storage,
)
from app.core.config import cost_settings
from app.core.database import get_engine, get_session_factory
from app.core.lifecycle import get_arq_redis_settings
from app.factories.document_processor import DocumentProcessorFactory
from app.services.documents import DocumentService
from app.services.page_extraction import PageStrategyRunner
from app.services.tool_call_extractor import ToolCallExtractor

logger = logging.getLogger(__name__)


async def startup(ctx: dict) -> None:
    ctx["session_factory"] = get_session_factory()
    ctx["llm_client"] = get_llm_client()
    ctx["storage"] = get_storage()
    ctx["classifier"] = get_classifier_service(ctx["llm_client"])
    ctx["tool_extractor"] = ToolCallExtractor(llm_client=ctx["llm_client"])
    ctx["page_runner"] = PageStrategyRunner(
        text_extractor=ctx["tool_extractor"]
    )
    ctx["processor_factory"] = DocumentProcessorFactory(
        classifier=ctx["classifier"],
        page_runner=ctx["page_runner"],
        storage=ctx["storage"],
        model=ctx["tool_extractor"].model,
    )
    logger.info("Worker services initialized.")


async def shutdown(ctx: dict) -> None:
    engine = get_engine()
    await engine.dispose()
    logger.info("Worker database connection closed.")


async def extraction_job(
    ctx: dict,
    document_id: str,
    doc_type: str | None = None,
    strategy: str | None = None,
) -> None:
    session_factory = ctx["session_factory"]
    processor_factory = ctx["processor_factory"]

    if isinstance(document_id, str):
        doc_uuid = UUID(document_id)
    else:
        doc_uuid = document_id

    async with session_factory() as db:
        try:
            logger.info("Starting extraction job for document %s", doc_uuid)
            documents = DocumentService(
                db=db,
                processor=processor_factory.create(db),
                processor_factory=processor_factory,
                arq_pool=None,
                session_factory=session_factory,
                price_for_model=cost_settings.price_for_model,
            )
            await documents._process_existing_inline(
                document_id=doc_uuid,
                doc_type=doc_type,
                strategy=strategy,
            )
            logger.info(
                "Successfully finished extraction job for document %s",
                doc_uuid,
            )

        except Exception as exc:
            logger.exception(
                "Background extraction job failed for document %s: %s",
                doc_uuid,
                exc,
            )
            raise


redis_settings = get_arq_redis_settings()


class WorkerSettings:
    functions = [extraction_job]
    max_jobs = 5
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = redis_settings
