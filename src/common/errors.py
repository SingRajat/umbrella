"""Typed exception hierarchy mapped to HTTP status codes."""

class UmbrellaError(Exception):
    """Base exception for all Umbrella errors."""
    def __init__(self, message: str, status_code: int = 500, error_code: str = "INTERNAL_SERVER_ERROR"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class IngestionError(UmbrellaError):
    """Raised when document ingestion fails (parsing, formatting, extraction)."""
    def __init__(self, message: str, status_code: int = 422, error_code: str = "INGESTION_FAILED"):
        super().__init__(message, status_code=status_code, error_code=error_code)


class FileTooLargeError(UmbrellaError):
    """Raised when uploaded file exceeds the allowed size limit."""
    def __init__(self, message: str = "File size exceeds 25MB limit", status_code: int = 413, error_code: str = "FILE_TOO_LARGE"):
        super().__init__(message, status_code=status_code, error_code=error_code)


class UnsupportedFileTypeError(UmbrellaError):
    """Raised when an unsupported file type is uploaded."""
    def __init__(self, message: str = "Unsupported file type", status_code: int = 400, error_code: str = "UNSUPPORTED_FILE_TYPE"):
        super().__init__(message, status_code=status_code, error_code=error_code)


class DuplicateDocumentError(UmbrellaError):
    """Raised when a document with identical SHA-256 hash already exists."""
    def __init__(self, message: str, existing_doc_id: str, status_code: int = 409, error_code: str = "DUPLICATE_DOCUMENT"):
        super().__init__(message, status_code=status_code, error_code=error_code)
        self.existing_doc_id = existing_doc_id


class RetrievalError(UmbrellaError):
    """Raised when vector retrieval fails."""
    def __init__(self, message: str, status_code: int = 500, error_code: str = "RETRIEVAL_FAILED"):
        super().__init__(message, status_code=status_code, error_code=error_code)


class GenerationError(UmbrellaError):
    """Raised when LLM answer generation fails."""
    def __init__(self, message: str, status_code: int = 502, error_code: str = "GENERATION_FAILED"):
        super().__init__(message, status_code=status_code, error_code=error_code)


class ValidationError(UmbrellaError):
    """Raised when request payload or output validation fails."""
    def __init__(self, message: str, status_code: int = 400, error_code: str = "VALIDATION_FAILED"):
        super().__init__(message, status_code=status_code, error_code=error_code)


class DocumentNotFoundError(UmbrellaError):
    """Raised when a requested document ID does not exist."""
    def __init__(self, message: str = "Document not found", status_code: int = 404, error_code: str = "DOCUMENT_NOT_FOUND"):
        super().__init__(message, status_code=status_code, error_code=error_code)


class StorageConnectionError(UmbrellaError):
    """Raised when ChromaDB or storage connection fails after retries."""
    def __init__(self, message: str = "Storage service unavailable", status_code: int = 503, error_code: str = "STORAGE_UNAVAILABLE"):
        super().__init__(message, status_code=status_code, error_code=error_code)
