# Architecture

```text
Employee / HR client
        |
        v
 FastAPI routes ---- request validation, guardrails, safe HTTP errors
        |
        +---- Upload -> allow-list validation -> Azure Blob (source)
        |                                      |
        |                             Document Intelligence Layout
        |                              PDF / DOCX / PPTX
        |                                      |
        |                         UTF-8 / CSV local readers
        |                                      |
        |                    Azure Blob (extracted blocks)
        |                                      |
        |               structure-aware chunking + metadata
        |                                      |
        |               embeddings -> Azure AI Search
        |
        +---- Query / Chat -> LangGraph workflow
        |                       |
        |            rewrite / HyDE / multi-query
        |                       |
        |         vector / BM25 / hybrid Search
        |                       |
        |         threshold -> score or MMR reranking
        |                       |
        |                 capped context
        |                       |
        |         Azure OpenAI generation
        |                       |
        |      citation + groundedness checks
        |
        +---- Evaluate -> existing Query service -> RAGAS metrics
```

The API layer validates request contracts and maps domain errors to safe HTTP responses. `IngestionService` coordinates storage and extraction, while the parser factory selects Document Intelligence for PDF/DOCX/PPTX and local readers for TXT/CSV. `AzureSearchClient` owns index schema, embedding upserts, filtering, and vector, keyword, or hybrid retrieval. Stable chunk IDs and merge-or-upload semantics allow re-indexing without duplicates.

The LangGraph state carries query options and retrieved documents through query preparation, retrieval, threshold filtering, reranking, capped context assembly, answer generation, and output validation. Score reranking preserves Azure Search ordering. MMR computes query and candidate embeddings to diversify the top results. Query responses retain every chunk supplied to the answer model so citation numbers and source array positions stay aligned.

`POST /api/v1/evaluate` invokes the existing RAG query service for each labeled question, then runs RAGAS metrics over the resulting answers and source contexts. It uses the configured Azure OpenAI models. Evaluation batches are intentionally small and can consume additional model quota.

Ingestion and evaluation run synchronously at the API boundary. A production deployment should consider a task queue, authentication/authorization, and stronger independently validated safety controls. Configurable input screening and a model-based groundedness check are defense-in-depth measures, not complete security guarantees.
