import json

import pytest
from langchain_core.documents import Document

from app.models.request import ChunkRequest
from app.services.ingestion_service import IngestionService


def test_chunk_request_defaults_metadata_safely():
    request = ChunkRequest(
        document_id="doc-1",
        text_blob_name="extracted/doc-1.json",
    )
    assert request.category == "hr_policy"
    assert request.subcategory is None
    assert request.policy_id is None


@pytest.mark.asyncio
async def test_chunk_copies_supplied_metadata_to_every_record(monkeypatch):
    import app.rag.chunking as chunking
    import app.services.ingestion_service as service_module

    stored = {}
    extracted = [{
        "content": "Insurance coverage details.",
        "file_name": "benefits.pdf",
        "page": 2,
        "source_blob": "sources/doc-1/benefits.pdf",
    }]
    monkeypatch.setattr(
        service_module, "download_blob", lambda _name: json.dumps(extracted).encode()
    )
    monkeypatch.setattr(
        service_module,
        "upload_blob",
        lambda name, data, _content_type: stored.update({name: data}),
    )
    monkeypatch.setattr(
        chunking,
        "split_documents",
        lambda documents: [Document(page_content=documents[0].page_content, metadata=documents[0].metadata)],
    )

    result = await IngestionService().chunk(
        "doc-1",
        "extracted/doc-1.json",
        category="benefits",
        subcategory="insurance",
        policy_id="HR-BEN-2026",
    )

    records = json.loads(stored[result["chunk_blob_name"]])
    assert result["chunk_count"] == 1
    assert records[0]["category"] == "benefits"
    assert records[0]["subcategory"] == "insurance"
    assert records[0]["policy_id"] == "HR-BEN-2026"
    assert records[0]["file_name"] == "benefits.pdf"
    assert records[0]["page"] == 2


@pytest.mark.asyncio
async def test_search_applies_metadata_filters_in_each_mode(monkeypatch):
    import app.integrations.azure_search as azure_search

    captured = {}

    class FakeEmbeddings:
        def embed_query(self, _query):
            return [0.1, 0.2]

    class FakeSearchClient:
        def __init__(self, *_args):
            pass

        def search(self, **kwargs):
            captured.update(kwargs)
            return [{
                "chunk_id": "chunk-1",
                "content": "Insurance policy",
                "category": "benefits",
                "subcategory": "insurance",
                "policy_id": "HR-BEN-2026",
                "@search.score": 1.0,
            }]

    monkeypatch.setattr(azure_search, "SearchClient", FakeSearchClient)
    monkeypatch.setattr(azure_search, "get_embeddings", FakeEmbeddings)
    monkeypatch.setattr(azure_search.settings, "azure_search_endpoint", "https://search.test")
    monkeypatch.setattr(azure_search.settings, "azure_search_api_key", "test")
    filters = {
        "category": "benefits",
        "subcategory": "insurance",
        "policy_id": "HR-BEN-2026",
    }
    expected = (
        "category eq 'benefits' and subcategory eq 'insurance' "
        "and policy_id eq 'HR-BEN-2026'"
    )

    for strategy in ("vector", "keyword", "hybrid"):
        result = await azure_search.AzureSearchClient().search(
            "insurance coverage",
            filters=filters,
            strategy=strategy,
        )
        assert captured["filter"] == expected
        assert {"subcategory", "policy_id"}.issubset(captured["select"])
        assert result[0]["subcategory"] == "insurance"
        assert result[0]["policy_id"] == "HR-BEN-2026"


@pytest.mark.asyncio
async def test_index_schema_and_upsert_include_policy_metadata(monkeypatch):
    import app.integrations.azure_search as azure_search

    captured = {"index_calls": 0}

    class FakeIndexClient:
        def __init__(self, *_args):
            pass

        def create_or_update_index(self, index):
            captured["index_calls"] += 1
            captured["fields"] = {field.name: field for field in index.fields}

    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[0.1, 0.2] for _ in texts]

    class FakeSearchClient:
        def __init__(self, *_args):
            pass

        def merge_or_upload_documents(self, docs):
            captured["docs"] = docs
            return [type("Result", (), {"succeeded": True})() for _ in docs]

        def search(self, **_kwargs):
            return []

    monkeypatch.setattr(azure_search, "SearchIndexClient", FakeIndexClient)
    monkeypatch.setattr(azure_search, "SearchClient", FakeSearchClient)
    monkeypatch.setattr(azure_search, "get_embeddings", FakeEmbeddings)
    monkeypatch.setattr(azure_search.settings, "azure_search_endpoint", "https://search.test")
    monkeypatch.setattr(azure_search.settings, "azure_search_api_key", "test")

    count = await azure_search.AzureSearchClient().index_chunks([{
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "content": "Insurance policy",
        "category": "benefits",
        "subcategory": "insurance",
        "policy_id": "HR-BEN-2026",
    }])

    assert count == 1
    assert captured["index_calls"] == 1
    for field_name in ("category", "subcategory", "policy_id"):
        assert captured["fields"][field_name].filterable is True
        assert captured["docs"][0][field_name] == {
            "category": "benefits",
            "subcategory": "insurance",
            "policy_id": "HR-BEN-2026",
        }[field_name]
