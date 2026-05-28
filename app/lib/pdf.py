from dataclasses import dataclass

import pymupdf

from app.core.exceptions import PDFParseError


@dataclass(frozen=True)
class ParsedPDF:
    pages_text: list[str]
    page_count: int


def parse_pdf(
    file_content: bytes, filename: str = "document.pdf"
) -> ParsedPDF:
    try:
        doc = pymupdf.open(stream=file_content, filetype="pdf")
    except Exception as exc:
        raise PDFParseError(filename, exc) from exc

    try:
        if doc.page_count == 0:
            raise PDFParseError(
                filename, RuntimeError("PDF does not contain any pages")
            )

        pages_text = []
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                pages_text.append(page.get_text() or "")
            except Exception as exc:
                raise PDFParseError(
                    filename,
                    RuntimeError(f"Failed to extract page {page_num}: {exc}"),
                ) from exc

        return ParsedPDF(pages_text=pages_text, page_count=doc.page_count)
    finally:
        doc.close()


def first_page_text(
    file_content: bytes, filename: str = "document.pdf"
) -> str:
    try:
        doc = pymupdf.open(stream=file_content, filetype="pdf")
    except Exception as exc:
        raise PDFParseError(filename, exc) from exc

    try:
        if doc.page_count == 0:
            return ""
        return doc[0].get_text() or ""
    except Exception as exc:
        raise PDFParseError(
            filename, RuntimeError(f"Failed to extract first page: {exc}")
        ) from exc
    finally:
        doc.close()
