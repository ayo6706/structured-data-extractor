from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlmodel import select

from app.models.document import Document, DocumentStatus


class DocumentRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, document: Document) -> Document:
        self.db.add(document)
        await self.db.flush()
        return document

    async def get_by_id(self, document_id: UUID) -> Document | None:
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def set_status(
        self,
        document: Document,
        status: DocumentStatus,
    ) -> None:
        document.status = status
        self.db.add(document)
        await self.db.flush()

    async def get_with_extractions(
        self,
        document_id: UUID,
    ) -> Document | None:
        stmt = (
            select(Document)
            .options(selectinload(Document.extractions))
            .where(Document.id == document_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
