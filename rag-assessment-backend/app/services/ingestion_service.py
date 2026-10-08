import asyncio
import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.core.config import settings
from app.core.exceptions import InvalidDocumentError
from app.ingestion.parsers import get_parser
from app.ingestion.validation import validate_upload
from app.integrations.azure_blob import download_blob, upload_blob
from app.integrations.azure_search import AzureSearchClient

logger = logging.getLogger(__name__)


def stable_chunk_id(document_id: str, index: int, content: str) -> str:
    """Build a repeatable search key for unchanged document chunk order/content."""
    return hashlib.sha256(f"{document_id}:{index}:{content}".encode("utf-8")).hexdigest()


class IngestionService:
    """Coordinates source upload, extraction, chunking and search indexing."""

    async def upload(self, filename: str, content_type: str, data: bytes) -> dict:
        document_type = validate_upload(filename, content_type, data)
        document_id = str(uuid.uuid4())
        safe_name = Path(filename).name.replace("..", "_")
        blob_name = f"sources/{document_id}/{safe_name}"
        await asyncio.to_thread(upload_blob, blob_name, data, content_type)
        logger.info("document uploaded document_id=%s type=%s", document_id, document_type)
        return {
            "document_id": document_id,
            "file_name": safe_name,
            "status": "uploaded",
            "blob_name": blob_name,
            "content_type": content_type,
        }

    async def process(self, document_id: str, blob_name: str) -> dict:
        if not blob_name.startswith(f"sources/{document_id}/"):
            raise InvalidDocumentError("The source blob does not belong to this document ID.")
        raw = await asyncio.to_thread(download_blob, blob_name)
        filename = Path(blob_name).name
        blocks = await asyncio.to_thread(get_parser(filename).extract, raw)
        blocks = [block for block in blocks if block.get("content", "").strip()]
        if not blocks:
            raise InvalidDocumentError("The document contains no extractable text.")
        for block in blocks:
            block["file_name"] = filename
            block["document_type"] = Path(filename).suffix.lstrip(".").lower()
            block["source_blob"] = blob_name
        text_blob_name = f"extracted/{document_id}.json"
        payload = json.dumps(blocks, ensure_ascii=False).encode("utf-8")
        await asyncio.to_thread(upload_blob, text_blob_name, payload, "application/json")
        logger.info("document processed document_id=%s blocks=%d", document_id, len(blocks))
        return {
            "document_id": document_id,
            "status": "processed",
            "text_blob_name": text_blob_name,
            "character_count": sum(len(block["content"]) for block in blocks),
            "file_type": Path(filename).suffix.lstrip("."),
        }

    async def chunk(
        self,
        document_id: str,
        text_blob_name: str,
        file_name: str = "",
        category: str = "hr_policy",
        subcategory: str | None = None,
        policy_id: str | None = None,
    ) -> dict:
        if text_blob_name != f"extracted/{document_id}.json":
            raise InvalidDocumentError("The extracted-text blob does not match this document ID.")
        raw = await asyncio.to_thread(download_blob, text_blob_name)
        try:
            blocks = json.loads(raw.decode("utf-8"))
            if not isinstance(blocks, list):
                raise ValueError("expected block list")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise InvalidDocumentError("Processed text is malformed.") from exc
        documents = [
            Document(
                page_content=block["content"],
                metadata={key: value for key, value in block.items() if key != "content"},
            )
            for block in blocks
            if isinstance(block, dict) and isinstance(block.get("content"), str)
        ]
        from app.rag.chunking import split_documents

        chunks = await asyncio.to_thread(split_documents, documents)
        if not chunks:
            raise InvalidDocumentError("Processed text contains no usable content.")
        records = []
        for index, chunk in enumerate(chunks):
            stable = stable_chunk_id(document_id, index, chunk.page_content)
            chunk_metadata = dict(chunk.metadata)
            resolved_file_name = file_name or chunk_metadata.get("file_name") or document_id
            record = {
                **chunk_metadata,
                "chunk_id": stable,
                "document_id": document_id,
                "file_name": resolved_file_name,
                "document_type": Path(resolved_file_name).suffix.lstrip(".").lower(),
                "category": category,
                "subcategory": subcategory,
                "policy_id": policy_id,
                "content": chunk.page_content,
                "source_blob": (
                    chunk_metadata.get("source_blob")
                    or (
                        f"sources/{document_id}/{resolved_file_name}"
                        if file_name
                        else f"sources/{document_id}"
                    )
                ),
                "text_blob": text_blob_name,
            }
            records.append(record)
        chunk_blob = f"chunks/{document_id}.json"
        await asyncio.to_thread(
            upload_blob,
            chunk_blob,
            json.dumps(records, ensure_ascii=False).encode(),
            "application/json",
        )
        return {
            "document_id": document_id,
            "status": "chunked",
            "chunk_blob_name": chunk_blob,
            "chunk_count": len(records),
            "chunk_size": settings.rag_chunk_size,
            "chunk_overlap": settings.rag_chunk_overlap,
            "strategy": "recursive_character_with_source_blocks",
        }

    async def embed(self, document_id: str, chunk_blob_name: str) -> dict:
        if chunk_blob_name != f"chunks/{document_id}.json":
            raise InvalidDocumentError("The chunk blob does not match this document ID.")
        raw = await asyncio.to_thread(download_blob, chunk_blob_name)
        try:
            records = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidDocumentError("Chunk data is malformed.") from exc
        if not isinstance(records, list) or not records:
            raise InvalidDocumentError("No chunks are available to index.")
        count = await AzureSearchClient().index_chunks(records)
        return {"document_id": document_id, "status": "indexed", "indexed_chunks": count}


ingestion_service = IngestionService()
