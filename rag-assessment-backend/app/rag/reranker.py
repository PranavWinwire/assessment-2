import asyncio
import logging
import math
from typing import Any

from app.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


class Reranker:
    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
        strategy: str = "score",
        mmr_lambda: float = 0.7,
    ) -> list[dict[str, Any]]:
        if strategy == "score" or len(documents) < 2:
            return sorted(
                documents,
                key=lambda item: float(item.get("score", 0.0)),
                reverse=True,
            )[:top_k]

        try:
            embedder = get_embeddings()
            query_vector, document_vectors = await asyncio.gather(
                asyncio.to_thread(embedder.embed_query, query),
                asyncio.to_thread(
                    embedder.embed_documents,
                    [str(item.get("content", "")) for item in documents],
                ),
            )
        except Exception as exc:
            logger.exception("MMR embedding failed; falling back to score ordering")
            return sorted(
                documents,
                key=lambda item: float(item.get("score", 0.0)),
                reverse=True,
            )[:top_k]

        query_relevance = [_cosine(query_vector, vector) for vector in document_vectors]
        remaining = set(range(len(documents)))
        selected: list[int] = []
        while remaining and len(selected) < min(top_k, len(documents)):
            best_index = max(
                remaining,
                key=lambda index: (
                    mmr_lambda * query_relevance[index]
                    - (1 - mmr_lambda)
                    * max(
                        (_cosine(document_vectors[index], document_vectors[chosen])
                         for chosen in selected),
                        default=0.0,
                    )
                ),
            )
            selected.append(best_index)
            remaining.remove(best_index)
        logger.info(
            "MMR reranking completed candidates=%d selected=%d lambda=%.2f",
            len(documents),
            len(selected),
            mmr_lambda,
        )
        return [documents[index] for index in selected]
