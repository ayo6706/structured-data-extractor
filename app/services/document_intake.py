import logging
from dataclasses import dataclass
from uuid import uuid4

import anyio

from app.infrastructure.storage import StorageBackend
from app.lib.pdf import parse_pdf
from app.models.document import Document, DocumentStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentIntakeResult:
    document: Document
    pages_text: list[str]


class DocumentIntakeService:
    def __init__(self, *, storage: StorageBackend) -> None:
        self.storage = storage

    async def prepare_document(
        self,
        *,
        content: bytes,
        filename: str,
        status: DocumentStatus,
    ) -> DocumentIntakeResult:
        document_id = uuid4()
        parsed_pdf = await anyio.to_thread.run_sync(
            parse_pdf, content, filename
        )
        file_path = await self.storage.save(
            file_id=str(document_id),
            content=content,
            filename=filename,
        )
        document = Document(
            id=document_id,
            filename=filename,
            file_path=file_path,
            file_size_bytes=len(content),
            page_count=parsed_pdf.page_count,
            status=status,
        )
        return DocumentIntakeResult(
            document=document,
            pages_text=parsed_pdf.pages_text,
        )

    async def load_document(self, document: Document) -> list[str]:
        content = await self.storage.load(document.file_path)
        parsed_pdf = await anyio.to_thread.run_sync(
            parse_pdf, content, document.filename
        )
        return parsed_pdf.pages_text

    async def delete_upload(self, file_path: str) -> None:
        try:
            await self.storage.delete(file_path)
        except Exception as exc:
            logger.error(
                "Failed to delete orphaned upload %s: %s",
                file_path,
                exc,
            )
