class AppError(Exception):
    """Base class for expected application failures."""


class UnsupportedFileTypeError(AppError):
    """The uploaded extension or media type is not supported."""


class InvalidDocumentError(AppError):
    """A supported document is empty, corrupt, or unreadable."""


class DocumentNotFoundError(AppError):
    """A referenced source or extracted document does not exist."""


class StorageError(AppError):
    """Blob Storage operation failed."""


class SearchError(AppError):
    """Azure AI Search operation failed."""


class DocumentIntelligenceError(AppError):
    """Azure Document Intelligence processing failed."""


class ModelServiceError(AppError):
    """Embedding or chat model operation failed."""


class GuardrailBlockedError(AppError):
    """The request was rejected by an input safety check."""

    def __init__(self, message: str, reason: str = "blocked"):
        super().__init__(message)
        self.reason = reason


class EvaluationServiceError(AppError):
    """RAG evaluation could not be completed."""
