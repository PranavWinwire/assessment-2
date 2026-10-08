import csv
import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.core.exceptions import InvalidDocumentError, UnsupportedFileTypeError
from app.integrations.document_intelligence import analyze_layout


class DocumentParser(ABC):
    """Small interface that keeps extraction independent from API routes."""

    @abstractmethod
    def extract(self, data: bytes) -> list[dict[str, Any]]:
        """Return normalized content blocks with optional source metadata."""


class TextParser(DocumentParser):
    def extract(self, data: bytes) -> list[dict[str, Any]]:
        try:
            text = data.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError("Text document encoding is not supported.") from exc
        return [{"content": text}] if text else []


class CsvParser(DocumentParser):
    def extract(self, data: bytes) -> list[dict[str, Any]]:
        try:
            rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        except (UnicodeDecodeError, csv.Error) as exc:
            raise InvalidDocumentError("CSV document could not be read.") from exc
        if not rows:
            return []
        headers = list(rows[0])
        text = "\n".join(
            " | ".join(f"{key}: {row.get(key, '')}" for key in headers)
            for row in rows
        )
        return [{"content": f"Columns: {', '.join(headers)}\n{text}"}]


class DocumentIntelligenceParser(DocumentParser):
    """Use the Layout model for PDF and Office document text extraction."""

    def __init__(self, file_type: str):
        self.file_type = file_type

    def extract(self, data: bytes) -> list[dict[str, Any]]:
        blocks = analyze_layout(data)
        for block in blocks:
            page = block.pop("page", None)
            if self.file_type == ".pptx":
                if page is not None:
                    block["slide"] = page
            elif page is not None:
                block["page"] = page
        return blocks


def get_parser(filename: str) -> DocumentParser:
    """Select Document Intelligence or a small text/CSV reader by extension."""
    extension = Path(filename).suffix.lower()
    parsers = {
        ".pdf": lambda: DocumentIntelligenceParser(extension),
        ".docx": lambda: DocumentIntelligenceParser(extension),
        ".pptx": lambda: DocumentIntelligenceParser(extension),
        ".txt": TextParser,
        ".csv": CsvParser,
    }
    parser_factory = parsers.get(extension)
    if parser_factory is None:
        raise UnsupportedFileTypeError("Unsupported document type.")
    return parser_factory()
