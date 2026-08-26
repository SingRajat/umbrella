"""Typed exception hierarchy for the Umbrella RAG system."""


class UmbrellaError(Exception):
    """Base exception for all domain-specific errors in Umbrella."""

    def __init__(self, message: str, error_code: str = "INTERNAL_ERROR", status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code


class IngestionError(UmbrellaError):
    """Raised when document loading, cleaning, chunking, or indexing fails."""

    def __init__(self, message: str, error_code: str = "INGESTION_ERROR", status_code: int = 422):
        super().__init__(message, error_code=error_code, status_code=status_code)


class RetrievalError(UmbrellaError):
    """Raised when vector search or document retrieval fails."""

    def __init__(self, message: str, error_code: str = "RETRIEVAL_ERROR", status_code: int = 500):
        super().__init__(message, error_code=error_code, status_code=status_code)


class GenerationError(UmbrellaError):
    """Raised when LLM invocation or prompt formatting fails."""

    def __init__(self, message: str, error_code: str = "GENERATION_ERROR", status_code: int = 502):
        super().__init__(message, error_code=error_code, status_code=status_code)


class ValidationError(UmbrellaError):
    """Raised when request payload or output verification fails."""

    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR", status_code: int = 400):
        super().__init__(message, error_code=error_code, status_code=status_code)
