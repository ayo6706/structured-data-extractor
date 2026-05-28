import pymupdf
import pytest

from app.core.exceptions import PDFParseError
from app.lib.pdf import first_page_text, parse_pdf


def _create_test_pdf(pages_text: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:  # Only write text if it's not empty
            page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def _create_zero_page_pdf() -> bytes:
    return (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [] /Count 0 >> endobj\n"
        b"xref\n"
        b"0 3\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"trailer << /Root 1 0 R /Size 3 >>\n"
        b"startxref\n"
        b"111\n"
        b"%%EOF\n"
    )


def test_parse_success() -> None:
    expected_texts = ["Page 1 Content", "Page 2 Content", ""]
    pdf_bytes = _create_test_pdf(expected_texts)

    parsed_pdf = parse_pdf(pdf_bytes, "test.pdf")

    assert parsed_pdf.page_count == 3
    assert len(parsed_pdf.pages_text) == 3
    assert "Page 1 Content" in parsed_pdf.pages_text[0]
    assert "Page 2 Content" in parsed_pdf.pages_text[1]
    assert parsed_pdf.pages_text[2] == ""


def test_parse_corrupted_pdf() -> None:
    corrupted_bytes = b"not a pdf at all"

    with pytest.raises(PDFParseError) as exc_info:
        parse_pdf(corrupted_bytes, "corrupted.pdf")

    assert "Failed to parse PDF at corrupted.pdf" in str(exc_info.value)


def test_parse_rejects_zero_page_pdf() -> None:
    with pytest.raises(PDFParseError) as exc_info:
        parse_pdf(_create_zero_page_pdf(), "empty.pdf")

    assert "PDF does not contain any pages" in str(exc_info.value)


def test_first_page_text() -> None:
    expected_texts = ["First page text", "Second page text"]
    pdf_bytes = _create_test_pdf(expected_texts)

    first_text = first_page_text(pdf_bytes, "test.pdf")
    assert "First page text" in first_text
    assert "Second page text" not in first_text


def test_first_page_text_empty_pdf() -> None:
    pdf_bytes = _create_test_pdf([""])

    first_text = first_page_text(pdf_bytes, "empty.pdf")
    assert first_text == ""
