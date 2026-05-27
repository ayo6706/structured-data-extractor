class ExtractorError(Exception):
    """Base exception for all domain errors."""


class UnsupportedDocumentTypeError(ExtractorError):
    """Raised when the document classifier returns 'unknown'."""

    def __init__(
        self, document_type: str, message: str | None = None
    ) -> None:
        self.document_type = document_type
        super().__init__(
            message or f"Unsupported document type: {document_type}"
        )


class PDFParseError(ExtractorError):
    """Raised when PyMuPDF fails to extract text."""

    def __init__(
        self, file_path: str, original_exception: Exception
    ) -> None:
        self.file_path = file_path
        self.original_exception = original_exception
        super().__init__(
            f"Failed to parse PDF at {file_path}: {original_exception}"
        )


class StorageError(ExtractorError):
    """Raised when file save or load operations fail."""

    def __init__(
        self, path: str, operation: str, original_exception: Exception
    ) -> None:
        self.path = path
        self.operation = operation
        self.original_exception = original_exception
        super().__init__(
            f"Failed to {operation} storage path {path}: "
            f"{original_exception}"
        )


class ClassificationError(ExtractorError):
    """Raised when the LLM classification call fails."""

    def __init__(
        self, message: str, original_exception: Exception | None = None
    ) -> None:
        self.original_exception = original_exception
        super().__init__(message)

