import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.core.exceptions import (
    ClassificationError,
    CorrectionValidationError,
    DocumentNotFoundError,
    ExtractionError,
    ExtractionNotFoundError,
    InvalidBatchError,
    InvalidDocumentTypeError,
    InvalidUploadError,
    PDFParseError,
    StorageError,
    UnsupportedDocumentTypeError,
    UploadTooLargeError,
)
from app.schemas.registry import SchemaRegistry

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(
        DocumentNotFoundError,
        _document_not_found,
    )
    app.add_exception_handler(
        ExtractionNotFoundError,
        _extraction_not_found,
    )
    app.add_exception_handler(
        InvalidDocumentTypeError,
        _invalid_document_type,
    )
    app.add_exception_handler(InvalidUploadError, _bad_request)
    app.add_exception_handler(InvalidBatchError, _bad_request)
    app.add_exception_handler(UploadTooLargeError, _upload_too_large)
    app.add_exception_handler(PDFParseError, _pdf_parse_error)
    app.add_exception_handler(StorageError, _storage_error)
    app.add_exception_handler(ClassificationError, _classification_error)
    app.add_exception_handler(
        UnsupportedDocumentTypeError,
        _unsupported_document_type,
    )
    app.add_exception_handler(ExtractionError, _extraction_error)
    app.add_exception_handler(
        CorrectionValidationError,
        _correction_validation_error,
    )
    app.add_exception_handler(SQLAlchemyError, _database_error)


async def _document_not_found(
    _request: Request,
    _exc: Exception,
) -> JSONResponse:
    return _response(status.HTTP_404_NOT_FOUND, "Document not found")


async def _extraction_not_found(
    _request: Request,
    _exc: Exception,
) -> JSONResponse:
    return _response(status.HTTP_404_NOT_FOUND, "Extraction not found")


async def _invalid_document_type(
    _request: Request,
    exc: InvalidDocumentTypeError,
) -> JSONResponse:
    return _response(
        status.HTTP_400_BAD_REQUEST,
        (
            f"Invalid doc_type '{exc.document_type}'. "
            f"Available types: {exc.available_types}"
        ),
    )


async def _bad_request(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    return _response(status.HTTP_400_BAD_REQUEST, str(exc))


async def _upload_too_large(
    _request: Request,
    exc: UploadTooLargeError,
) -> JSONResponse:
    return _response(status.HTTP_413_CONTENT_TOO_LARGE, str(exc))


async def _pdf_parse_error(
    _request: Request,
    exc: PDFParseError,
) -> JSONResponse:
    logger.debug("PDF parsing failed: %s", exc)
    return _response(status.HTTP_400_BAD_REQUEST, "Could not parse PDF")


async def _storage_error(
    _request: Request,
    exc: StorageError,
) -> JSONResponse:
    logger.error("Storage operation failed: %s", exc)
    action = (
        "load document file from"
        if exc.operation == "load"
        else "save uploaded file to"
    )
    return _response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        f"Failed to {action} storage",
    )


async def _classification_error(
    _request: Request,
    exc: ClassificationError,
) -> JSONResponse:
    logger.error("Classification failed: %s", exc)
    return _response(
        status.HTTP_502_BAD_GATEWAY,
        "Classification service unavailable",
    )


async def _unsupported_document_type(
    _request: Request,
    _exc: UnsupportedDocumentTypeError,
) -> JSONResponse:
    return _response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        (
            "Could not determine the document type. "
            f"Supported types: {SchemaRegistry.list_types()}"
        ),
    )


async def _extraction_error(
    _request: Request,
    exc: ExtractionError,
) -> JSONResponse:
    logger.error("Extraction failed: %s", exc)
    return _response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Extraction failed",
    )


async def _correction_validation_error(
    _request: Request,
    exc: CorrectionValidationError,
) -> JSONResponse:
    return _response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        {"warnings": exc.warnings},
    )


async def _database_error(
    _request: Request,
    exc: SQLAlchemyError,
) -> JSONResponse:
    logger.error("Database operation failed: %s", exc)
    return _response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Database operation failed",
    )


def _response(status_code: int, detail: object) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})
