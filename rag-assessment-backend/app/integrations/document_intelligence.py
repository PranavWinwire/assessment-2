import io
import logging
from typing import Any

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from app.core.config import settings
from app.core.exceptions import DocumentIntelligenceError, InvalidDocumentError

logger = logging.getLogger(__name__)


def get_document_intelligence_client() -> DocumentIntelligenceClient:
    """Create a configured Document Intelligence client."""
    if (
        not settings.document_intelligence_endpoint
        or not settings.document_intelligence_key
    ):
        raise DocumentIntelligenceError("Document Intelligence is not configured.")
    return DocumentIntelligenceClient(
        endpoint=settings.document_intelligence_endpoint.rstrip("/"),
        credential=AzureKeyCredential(settings.document_intelligence_key),
    )


def analyze_layout(data: bytes) -> list[dict[str, Any]]:
    """Extract layout blocks and normalize location metadata for RAG ingestion."""
    try:
        client = get_document_intelligence_client()
        poller = client.begin_analyze_document(
            settings.document_intelligence_model,
            body=io.BytesIO(data),
        )
        result = poller.result()
    except DocumentIntelligenceError:
        raise
    except HttpResponseError as exc:
        provider_error = getattr(exc, "error", None)
        logger.error(
            "Document Intelligence returned an error status=%s provider_code=%s",
            exc.status_code,
            getattr(provider_error, "code", None),
        )
        if exc.status_code in {400, 415, 422}:
            raise InvalidDocumentError(
                "Document Intelligence could not read this supported file."
            ) from exc
        raise DocumentIntelligenceError(
            "Document Intelligence could not analyze this file."
        ) from exc
    except Exception as exc:
        logger.exception(
            "Document Intelligence transport or polling failed exception_type=%s",
            type(exc).__name__,
        )
        raise DocumentIntelligenceError(
            "Document Intelligence could not analyze this file."
        ) from exc

    grouped: dict[tuple[int | None, str], list[str]] = {}
    if result.paragraphs:
        current_section = ""
        for paragraph in result.paragraphs:
            text = (paragraph.content or "").strip()
            if not text:
                continue
            role = getattr(paragraph.role, "value", paragraph.role)
            role_name = str(role or "").lower().replace("_", "")
            if role_name in {"title", "sectionheading"}:
                current_section = text
            regions = paragraph.bounding_regions or []
            page_number = regions[0].page_number if regions else None
            grouped.setdefault((page_number, current_section), []).append(text)

    if not grouped and result.pages:
        for page in result.pages:
            lines = [line.content.strip() for line in (page.lines or []) if line.content]
            if lines:
                grouped[(page.page_number, "")] = lines

    blocks = [
        {"content": "\n".join(lines), "page": page, "section": section}
        for (page, section), lines in grouped.items()
    ]
    blocks.extend(_normalize_tables(result.tables or []))
    if not blocks and result.content and result.content.strip():
        blocks.append({"content": result.content.strip()})
    return blocks


def _normalize_tables(tables: list[Any]) -> list[dict[str, Any]]:
    """Render recognized tables as readable rows rather than SDK objects."""
    blocks = []
    for table in tables:
        matrix = [
            [""] * table.column_count for _ in range(table.row_count)
        ]
        for cell in table.cells:
            matrix[cell.row_index][cell.column_index] = cell.content.strip()
        rows = [" | ".join(value for value in row if value) for row in matrix]
        text = "\n".join(row for row in rows if row)
        if not text:
            continue
        regions = table.bounding_regions or []
        page_number = regions[0].page_number if regions else None
        blocks.append({
            "content": f"Table:\n{text}",
            "page": page_number,
            "section": "Table",
        })
    return blocks
