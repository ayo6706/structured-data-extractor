from dataclasses import dataclass
from typing import Annotated

from fastapi import File, Query, UploadFile

from app.core.config import get_app_settings
from app.core.exceptions import (
    InvalidBatchError,
    InvalidDocumentTypeError,
    InvalidUploadError,
    UploadTooLargeError,
)
from app.schemas.registry import SchemaRegistry

PDF_CONTENT_TYPE = "application/pdf"
UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_BATCH_SIZE = 20


@dataclass(frozen=True)
class UploadedDocument:
    filename: str
    content: bytes


async def valid_pdf_upload(file: UploadFile) -> UploadedDocument:
    if not _is_pdf_upload(file):
        await file.close()
        raise InvalidUploadError("Only PDF uploads are supported")

    return UploadedDocument(
        filename=file.filename or "uploaded.pdf",
        content=await _read_upload_file(
            file,
            max_size=get_app_settings().MAX_UPLOAD_SIZE_BYTES,
        ),
    )


async def optional_pdf_upload(
    file: UploadFile | None = File(default=None),
) -> UploadedDocument | None:
    if file is None:
        return None

    return await valid_pdf_upload(file)


async def valid_pdf_batch(
    files: list[UploadFile] = File(),
) -> list[UploadedDocument]:
    if not files:
        raise InvalidBatchError("No files uploaded")
    if len(files) > MAX_BATCH_SIZE:
        raise InvalidBatchError(f"Maximum batch size is {MAX_BATCH_SIZE} files")

    invalid_file = next(
        (file for file in files if not _is_pdf_upload(file)), None
    )
    if invalid_file is not None:
        raise InvalidUploadError(
            "Only PDF uploads are supported. "
            f"Invalid file: {invalid_file.filename}"
        )

    return [await valid_pdf_upload(file) for file in files]


def valid_doc_type(
    doc_type: Annotated[
        str | None,
        Query(description="Skip classification with a known document type"),
    ] = None,
) -> str | None:
    available_types = SchemaRegistry.list_types()
    if doc_type is not None and doc_type not in available_types:
        raise InvalidDocumentTypeError(doc_type, available_types)
    return doc_type


def _is_pdf_upload(file: UploadFile) -> bool:
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    return content_type == PDF_CONTENT_TYPE or filename.endswith(".pdf")


async def _read_upload_file(
    file: UploadFile,
    *,
    max_size: int,
    chunk_size: int = UPLOAD_CHUNK_SIZE,
) -> bytes:
    chunks = []
    total_size = 0
    while chunk := await file.read(chunk_size):
        total_size += len(chunk)
        if total_size > max_size:
            await file.close()
            raise UploadTooLargeError("Uploaded file is too large")
        chunks.append(chunk)
    return b"".join(chunks)
