from pathlib import Path

from app.core.config import settings
from app.core.exceptions import InvalidDocumentError, UnsupportedFileTypeError

SUPPORTED_MEDIA_TYPES = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
    },
    ".txt": {"text/plain", "application/octet-stream"},
    ".pptx": {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/octet-stream",
    },
    ".csv": {"text/csv", "application/csv", "application/octet-stream"},
}
SUPPORTED_EXTENSIONS = set(SUPPORTED_MEDIA_TYPES)
SIGNATURES = {
    ".pdf": b"%PDF-",
    ".docx": b"PK\x03\x04",
    ".pptx": b"PK\x03\x04",
}


def validate_upload(filename: str, content_type: str | None, data: bytes) -> str:
    """Validate extension, declared media type, size and recognizable content."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError("Supported file types are PDF, DOCX, TXT, PPTX, and CSV.")
    media_type = (content_type or "application/octet-stream").split(";")[0].lower()
    if media_type not in SUPPORTED_MEDIA_TYPES[extension]:
        raise UnsupportedFileTypeError(
            "The uploaded content type does not match a supported file type."
        )
    if not data:
        raise InvalidDocumentError("The uploaded document is empty.")
    if len(data) > settings.max_upload_bytes:
        raise InvalidDocumentError("The uploaded document exceeds the configured size limit.")
    signature = SIGNATURES.get(extension)
    if signature and not data.startswith(signature):
        raise InvalidDocumentError("The file content is not a valid supported document.")
    if extension in {".txt", ".csv"}:
        try:
            data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError("Text and CSV files must be UTF-8 encoded.") from exc
    return extension[1:]
