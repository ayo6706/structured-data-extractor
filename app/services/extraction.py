import pymupdf
from fastapi.concurrency import run_in_threadpool

from app.core.exceptions import (
    PDFParseError,
    UnsupportedDocumentTypeError,
)
from app.schemas.registry import SchemaRegistry
from app.services.classifier import ClassifierService


class ExtractService:
    def __init__(self, classifier: ClassifierService) -> None:
        self.classifier = classifier

    async def resolve_doc_type(
        self,
        *,
        content: bytes,
        filename: str,
        doc_type: str | None,
    ) -> str:
        if doc_type is not None:
            self._validate_doc_type(doc_type)
            return doc_type

        first_page_text = await run_in_threadpool(
            self._extract_first_page_text,
            content,
            filename,
        )
        resolved_type = await self.classifier.classify(first_page_text)

        if resolved_type == "unknown":
            raise UnsupportedDocumentTypeError(resolved_type)

        return resolved_type

    @staticmethod
    def _validate_doc_type(doc_type: str) -> None:
        if doc_type not in SchemaRegistry.list_types():
            raise UnsupportedDocumentTypeError(doc_type)

    @staticmethod
    def _extract_first_page_text(content: bytes, filename: str) -> str:
        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise PDFParseError(filename, exc) from exc

        try:
            return doc[0].get_text() if doc.page_count > 0 else ""
        finally:
            doc.close()
