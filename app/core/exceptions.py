class ExtractorError(Exception):
    """Base exception for all domain errors."""


class UnsupportedDocumentTypeError(ExtractorError):
    """Raised when the document classifier returns 'unknown'."""

    def __init__(self, document_type: str, message: str | None = None) -> None:
        self.document_type = document_type
        super().__init__(
            message or f"Unsupported document type: {document_type}"
        )


class PDFParseError(ExtractorError):
    """Raised when PyMuPDF fails to extract text."""

    def __init__(self, file_path: str, original_exception: Exception) -> None:
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
            f"Failed to {operation} storage path {path}: {original_exception}"
        )


class DocumentNotFoundError(ExtractorError):
    """Raised when a requested document row does not exist."""

    def __init__(self, document_id: object) -> None:
        self.document_id = document_id
        super().__init__(f"Document {document_id} not found")


class ExtractionNotFoundError(ExtractorError):
    """Raised when a requested extraction row does not exist."""

    def __init__(self, extraction_id: object) -> None:
        self.extraction_id = extraction_id
        super().__init__(f"Extraction {extraction_id} not found")


class InvalidDocumentTypeError(ExtractorError):
    def __init__(self, document_type: str, available_types: list[str]) -> None:
        self.document_type = document_type
        self.available_types = available_types
        super().__init__(f"Invalid document type: {document_type}")


class InvalidUploadError(ExtractorError):
    pass


class UploadTooLargeError(ExtractorError):
    pass


class InvalidBatchError(ExtractorError):
    pass


class CorrectionValidationError(ExtractorError):
    def __init__(self, warnings: list[str]) -> None:
        self.warnings = warnings
        super().__init__("Correction does not contain valid schema fields")


class ClassificationError(ExtractorError):
    """Raised when the LLM classification call fails."""

    def __init__(
        self, message: str, original_exception: Exception | None = None
    ) -> None:
        self.original_exception = original_exception
        super().__init__(message)


class ExtractionError(ExtractorError):
    """Raised when structured data extraction fails after retries."""

    def __init__(
        self,
        message: str,
        attempts: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        original_exception: Exception | None = None,
    ) -> None:
        self.attempts = attempts
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.original_exception = original_exception
        super().__init__(message)
