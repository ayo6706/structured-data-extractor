from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.storage import StorageBackend
from app.repositories.extractions import ExtractionRepository
from app.repositories.llm_usage import LLMUsageRepository
from app.services.audit import AuditRecorder
from app.services.classifier import ClassifierService
from app.services.document_processor import DocumentProcessor
from app.services.page_extraction import PageStrategyRunner


class DocumentProcessorFactory:
    def __init__(
        self,
        *,
        classifier: ClassifierService,
        page_runner: PageStrategyRunner,
        storage: StorageBackend,
        model: str,
    ) -> None:
        self.classifier = classifier
        self.page_runner = page_runner
        self.storage = storage
        self.model = model

    def create(self, _db: AsyncSession | None = None) -> DocumentProcessor:
        return DocumentProcessor(
            classifier=self.classifier,
            page_runner=self.page_runner,
        )

    def create_audit_recorder(self, db: AsyncSession) -> AuditRecorder:
        return AuditRecorder(
            db=db,
            extractions=ExtractionRepository(db),
            llm_usage=LLMUsageRepository(db),
            model=self.model,
        )
