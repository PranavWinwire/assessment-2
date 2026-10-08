import asyncio
import logging
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from app.core.config import settings
from app.core.exceptions import SearchError
from app.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)


class AzureSearchClient:
    """Azure AI Search index management, vector indexing and retrieval."""

    def __init__(self):
        self.endpoint = settings.azure_search_endpoint
        self.api_key = settings.azure_search_api_key
        self.index_name = settings.azure_search_index

    def _credential(self) -> AzureKeyCredential:
        if not self.endpoint or not self.api_key:
            raise SearchError("Azure AI Search is not configured.")
        return AzureKeyCredential(self.api_key)

    async def ensure_index(self) -> None:
        """Create the index with vector dimensions matching the embedding deployment."""
        def create() -> None:
            index_client = SearchIndexClient(self.endpoint, self._credential())
            fields = [
                SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="file_name", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="document_type", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="category", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="subcategory", type=SearchFieldDataType.String, filterable=True, nullable=True),
                SimpleField(name="policy_id", type=SearchFieldDataType.String, filterable=True, nullable=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SimpleField(
                    name="page",
                    type=SearchFieldDataType.Int32,
                    filterable=True,
                    nullable=True,
                ),
                SimpleField(
                    name="slide",
                    type=SearchFieldDataType.Int32,
                    filterable=True,
                    nullable=True,
                ),
                SearchableField(
                    name="section",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    nullable=True,
                ),
                SimpleField(name="source_blob", type=SearchFieldDataType.String),
                SimpleField(name="text_blob", type=SearchFieldDataType.String),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=settings.azure_embedding_dimensions,
                    vector_search_profile_name="hr-vector-profile",
                ),
            ]
            vector_search = VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="hr-hnsw")],
                profiles=[
                    VectorSearchProfile(
                        name="hr-vector-profile",
                        algorithm_configuration_name="hr-hnsw",
                    )
                ],
            )
            index_client.create_or_update_index(
                SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
            )
        try:
            await asyncio.to_thread(create)
        except Exception as exc:
            raise SearchError("Azure AI Search index could not be prepared.") from exc

    async def index_chunks(self, chunks: list[dict[str, Any]]) -> int:
        await self.ensure_index()
        try:
            texts = [chunk["content"] for chunk in chunks]
            vectors = await asyncio.to_thread(get_embeddings().embed_documents, texts)
            docs = [
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "file_name": chunk.get("file_name", ""),
                    "document_type": chunk.get("document_type", ""),
                    "category": chunk.get("category", ""),
                    "subcategory": chunk.get("subcategory"),
                    "policy_id": chunk.get("policy_id"),
                    "content": chunk["content"],
                    "content_vector": vector,
                    "page": chunk.get("page"),
                    "slide": chunk.get("slide"),
                    "section": chunk.get("section", ""),
                    "source_blob": chunk.get("source_blob", ""),
                    "text_blob": chunk.get("text_blob", ""),
                }
                for chunk, vector in zip(chunks, vectors)
            ]
            client = SearchClient(self.endpoint, self.index_name, self._credential())
            result = await asyncio.to_thread(client.merge_or_upload_documents, docs)
            failures = [item for item in result if not item.succeeded]
            if failures:
                raise SearchError("Some chunks could not be indexed.")
            document_ids = {chunk["document_id"] for chunk in chunks}
            indexed_ids = {chunk["chunk_id"] for chunk in chunks}
            for document_id in document_ids:
                escaped_id = document_id.replace("'", "''")
                existing = await asyncio.to_thread(
                    lambda: list(client.search(
                        search_text="*",
                        filter=f"document_id eq '{escaped_id}'",
                        select=["chunk_id"],
                    ))
                )
                stale_ids = [
                    item["chunk_id"]
                    for item in existing
                    if item["chunk_id"] not in indexed_ids
                ]
                if stale_ids:
                    await asyncio.to_thread(
                        client.delete_documents,
                        documents=[{"chunk_id": chunk_id} for chunk_id in stale_ids],
                    )
            return len(docs)
        except SearchError:
            raise
        except Exception as exc:
            raise SearchError("Chunks could not be embedded or indexed.") from exc

    async def search(
        self,
        query: str,
        top_k: int = 20,
        filters: dict | None = None,
        hybrid: bool = True,
        strategy: str | None = None,
        vector_weight: float = 1.0,
    ) -> list[dict]:
        """Run vector, lexical, or native hybrid search and normalize citations."""
        mode = strategy or ("hybrid" if hybrid else "vector")
        try:
            vector_query = None
            if mode in {"hybrid", "vector"}:
                vector = await asyncio.to_thread(get_embeddings().embed_query, query)
                vector_query = VectorizedQuery(
                    vector=vector,
                    k_nearest_neighbors=top_k,
                    fields="content_vector",
                    weight=vector_weight if mode == "hybrid" else 1.0,
                )
            clauses = []
            for key, value in (filters or {}).items():
                if key in {
                    "document_id", "document_type", "category", "subcategory", "policy_id", "file_name"
                }:
                    escaped = str(value).replace("'", "''")
                    clauses.append(f"{key} eq '{escaped}'")
            kwargs: dict[str, Any] = {
                "search_text": query if mode in {"hybrid", "keyword"} else None,
                "top": top_k,
                "filter": " and ".join(clauses) or None,
                "select": [
                    "chunk_id",
                    "document_id",
                    "file_name",
                    "document_type",
                    "category",
                    "subcategory",
                    "policy_id",
                    "content",
                    "page",
                    "slide",
                    "section",
                    "source_blob",
                    "text_blob",
                ],
            }
            if vector_query:
                kwargs["vector_queries"] = [vector_query]
            client = SearchClient(self.endpoint, self.index_name, self._credential())
            results = await asyncio.to_thread(lambda: list(client.search(**kwargs)))
            normalized = []
            for item in results:
                normalized.append({
                    **dict(item),
                    "chunk_id": item.get("chunk_id"),
                    "score": item.get("@search.score", 0.0),
                    "content": item.get("content", ""),
                })
            return normalized
        except SearchError:
            raise
        except Exception as exc:
            logger.exception("search failed strategy=%s", mode)
            raise SearchError("Azure AI Search query failed.") from exc
