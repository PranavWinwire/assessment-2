import asyncio
import logging

from langchain_core.exceptions import OutputParserException

from app.core.config import settings
from app.core.exceptions import ModelServiceError
from app.models.generated_answer import GeneratedAnswer, GroundednessVerdict
from app.graph.state import RAGState
from app.rag.citations import extract_sources
from app.rag.context import build_context, select_context_documents
from app.rag.hyde import generate_hypothetical_document
from app.rag.llm import invoke_structured_chat
from app.rag.multi_query import generate_queries
from app.rag.output_validation import FALLBACK_ANSWER, render_validated_answer
from app.rag.prompts import get_rag_prompt
from app.rag.prompts import GROUNDEDNESS_PROMPT
from app.rag.query_rewrite import rewrite_query
from app.rag.retriever import RAGRetriever
from app.rag.reranker import Reranker
from app.rag.guardrails import output_is_safe

logger = logging.getLogger(__name__)


async def prepare_query(state: RAGState) -> RAGState:
    if state.get("use_query_rewrite") and state.get("chat_history"):
        state["query"] = await rewrite_query(
            state["query"],
            state["chat_history"],
        )

    queries = [state["query"]]
    if state.get("use_hyde"):
        queries.append(await generate_hypothetical_document(state["query"]))
    if state.get("use_multi_query"):
        queries.extend(await generate_queries(state["query"]))
    state["retrieval_queries"] = list(dict.fromkeys(query for query in queries if query))
    return state


async def retrieve(state: RAGState) -> RAGState:
    retriever = RAGRetriever()
    strategy = state.get(
        "retrieval_strategy", "hybrid" if state.get("use_hybrid", True) else "vector"
    )
    batches = await asyncio.gather(*[
        retriever.retrieve(
            query=query,
            top_k=state.get("top_k", 5),
            candidate_k=state.get("candidate_k", 20),
            filters=state.get("filters", {}),
            hybrid=strategy != "keyword",
            strategy=strategy,
            vector_weight=state.get("hybrid_vector_weight", settings.rag_hybrid_vector_weight),
        )
        for query in state.get("retrieval_queries", [state["query"]])
    ])
    merged = {}
    for batch in batches:
        for document in batch:
            key = document.get("chunk_id")
            if key and (
                key not in merged
                or document.get("score", 0) > merged[key].get("score", 0)
            ):
                merged[key] = document
    documents = sorted(
        merged.values(), key=lambda item: item.get("score", 0), reverse=True
    )
    unique_documents = []
    seen_passages = set()
    for document in documents:
        passage = " ".join(str(document.get("content", "")).split()).casefold()
        identity = (
            str(document.get("file_name", "")).casefold(),
            str(document.get("page", "")),
            str(document.get("slide", "")),
            str(document.get("section", "")).casefold(),
            passage,
        )
        if identity in seen_passages:
            continue
        seen_passages.add(identity)
        unique_documents.append(document)
    if len(unique_documents) != len(documents):
        logger.info(
            "duplicate retrieval passages removed count=%d",
            len(documents) - len(unique_documents),
        )
    documents = unique_documents
    threshold = state.get("score_threshold", settings.rag_score_threshold)
    low_confidence_threshold = {
        "vector": settings.rag_low_confidence_vector_threshold,
        "keyword": settings.rag_low_confidence_keyword_threshold,
        "hybrid": settings.rag_low_confidence_hybrid_threshold,
    }.get(strategy, settings.rag_low_confidence_hybrid_threshold)
    best_score = max(
        (float(document.get("score", 0.0)) for document in documents),
        default=0.0,
    )
    state["low_confidence"] = best_score < low_confidence_threshold
    if state["low_confidence"]:
        logger.info(
            "low confidence retrieval strategy=%s score=%.4f threshold=%.4f",
            strategy,
            best_score,
            low_confidence_threshold,
        )
    state["documents"] = [
        document for document in documents
        if document.get("score", 0) >= threshold
    ]
    logger.info(
        "retrieval completed strategy=%s retrieved=%d retained=%d",
        strategy,
        len(merged),
        len(state["documents"]),
    )
    return state


async def rerank(state: RAGState) -> RAGState:
    docs = state.get("documents", [])
    if state.get("use_reranker"):
        state["documents"] = await Reranker().rerank(
            state["query"],
            docs,
            state.get("top_k", 5),
            strategy=state.get("reranker_strategy", "score"),
            mmr_lambda=state.get("mmr_lambda", settings.rag_mmr_lambda),
        )
    else:
        state["documents"] = docs[:state.get("top_k", 5)]
    return state


async def generate(state: RAGState) -> RAGState:
    if not state.get("documents"):
        state["answer"] = FALLBACK_ANSWER
        state["sources"] = []
        state["output_validated"] = True
        state["citation_validation"] = True
        state["groundedness_validation"] = True
        state["refusal"] = True
        state["refusal_reason"] = "no_context"
        return state
    context_documents = select_context_documents(state.get("documents", []))
    context = build_context(context_documents)
    prompt = get_rag_prompt(state.get("prompt_strategy", "zero_shot"))
    sources = extract_sources(context_documents)
    try:
        generated = await invoke_structured_chat(
            prompt,
            {"query": state["query"], "context": context},
            GeneratedAnswer,
        )
        state["output_validated"] = True
        answer = render_validated_answer(generated, len(sources))
        state["citation_validation"] = True
        is_abstention = "not have enough information" in generated.answer.casefold()
        if not output_is_safe(generated.answer):
            logger.warning("guardrail blocked generated output category=prompt_leak action=refuse")
            state["answer"] = FALLBACK_ANSWER
            state["sources"] = []
            state["output_validated"] = False
            state["groundedness_validation"] = False
            state["refusal"] = True
            state["refusal_reason"] = "unsafe_output"
            return state
        if not is_abstention and settings.rag_groundedness_check_enabled:
            try:
                verdict = await invoke_structured_chat(
                    GROUNDEDNESS_PROMPT,
                    {"context": context, "answer": generated.answer},
                    GroundednessVerdict,
                )
            except (OutputParserException, ModelServiceError):
                logger.exception("groundedness check failed; using safe fallback")
                state["answer"] = FALLBACK_ANSWER
                state["sources"] = []
                state["groundedness_validation"] = False
                state["refusal"] = True
                state["refusal_reason"] = "groundedness_check_failed"
                return state
            if not verdict.is_grounded:
                logger.warning("guardrail rejected ungrounded answer action=refuse")
                state["answer"] = FALLBACK_ANSWER
                state["sources"] = []
                state["groundedness_validation"] = False
                state["refusal"] = True
                state["refusal_reason"] = "ungrounded_answer"
                return state
        state["answer"] = answer
        state["groundedness_validation"] = is_abstention or settings.rag_groundedness_check_enabled
        # Keep every source in context so SOURCE N labels map to the same list
        # positions and evaluation can inspect all retrieved passages.
        state["sources"] = [] if is_abstention else sources
        state["refusal"] = is_abstention
        state["refusal_reason"] = "insufficient_context" if is_abstention else None
    except OutputParserException:
        logger.warning("answer output parsing failed; using safe fallback")
        state["answer"] = FALLBACK_ANSWER
        state["sources"] = []
        state["output_validated"] = False
        state["citation_validation"] = False
        state["groundedness_validation"] = False
        state["refusal"] = True
        state["refusal_reason"] = "output_parse_failure"
    except ValueError as exc:
        logger.warning("answer citation validation failed reason=%s; using safe fallback", str(exc))
        state["answer"] = FALLBACK_ANSWER
        state["sources"] = []
        state["output_validated"] = True
        state["citation_validation"] = False
        state["groundedness_validation"] = False
        state["refusal"] = True
        state["refusal_reason"] = "citation_validation_failure"
    return state
