import logging

from app.models.request import QueryRequest
from app.models.response import QueryResponse
from app.graph.workflow import rag_graph
from app.core.config import settings
from app.rag.guardrails import screen_query

logger = logging.getLogger(__name__)


class RAGService:
    @staticmethod
    def _retrieval_strategy(request: QueryRequest) -> str:
        return request.retrieval_strategy or (
            "hybrid" if request.use_hybrid else "vector"
        )

    async def query(
        self,
        request: QueryRequest,
        conversational: bool = False,
    ) -> QueryResponse:
        screen_query(request.query)
        logger.info(
            "query received conversational=%s strategy=%s",
            conversational,
            self._retrieval_strategy(request),
        )
        state = {
            "query": request.query,
            "original_query": request.query,
            "conversation_id": request.conversation_id,
            "chat_history": request.chat_history,
            "top_k": request.top_k,
            "candidate_k": request.candidate_k,
            "use_hybrid": request.use_hybrid,
            "retrieval_strategy": self._retrieval_strategy(request),
            "prompt_strategy": request.prompt_strategy,
            "use_reranker": request.use_reranker,
            "reranker_strategy": request.reranker_strategy,
            "mmr_lambda": request.mmr_lambda,
            "hybrid_vector_weight": request.hybrid_vector_weight,
            "use_query_rewrite": request.use_query_rewrite or conversational,
            "use_hyde": request.use_hyde,
            "use_multi_query": request.use_multi_query,
            "score_threshold": settings.rag_score_threshold,
            "filters": request.filters,
            "documents": [],
            "answer": "",
            "sources": [],
        }

        result = await rag_graph.ainvoke(state)

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            metadata={
                "retrieval_strategy": self._retrieval_strategy(request),
                "reranked": request.use_reranker,
                "reranker_strategy": request.reranker_strategy,
                "mmr_lambda": request.mmr_lambda,
                "hybrid_vector_weight": request.hybrid_vector_weight,
                "prompt_strategy": request.prompt_strategy,
                "query_rewrite": request.use_query_rewrite or conversational,
                "hyde": request.use_hyde,
                "multi_query": request.use_multi_query,
                "output_validated": result.get("output_validated", False),
                "citation_validation": result.get("citation_validation", False),
                "groundedness_validation": result.get("groundedness_validation", False),
                "refusal": result.get("refusal", False),
                "refusal_reason": result.get("refusal_reason"),
                "low_confidence": result.get("low_confidence", False),
                "retrieved_passage_count": len(result.get("documents", [])),
            },
        )


rag_service = RAGService()
import logging

