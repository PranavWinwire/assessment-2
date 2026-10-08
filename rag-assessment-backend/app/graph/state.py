from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    query: str
    original_query: str
    conversation_id: str | None
    chat_history: list[dict[str, str]]
    top_k: int
    candidate_k: int
    use_hybrid: bool
    retrieval_strategy: str
    prompt_strategy: str
    use_reranker: bool
    reranker_strategy: str
    mmr_lambda: float
    hybrid_vector_weight: float
    use_query_rewrite: bool
    use_hyde: bool
    use_multi_query: bool
    filters: dict[str, str]
    retrieval_queries: list[str]
    documents: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, Any]]
    score_threshold: float
    low_confidence: bool
    output_validated: bool
    citation_validation: bool
    groundedness_validation: bool
    refusal: bool
    refusal_reason: str | None
