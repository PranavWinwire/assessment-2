import asyncio
import functools
import importlib
import logging
import math
import sys
import threading
import types
from typing import Any

from app.core.exceptions import EvaluationServiceError
from app.models.request import EvaluationRequest, QueryRequest
from app.models.response import (
    EvaluationResponse,
    EvaluationResult,
)
from app.rag.embeddings import get_embeddings
from app.rag.llm import get_chat_model
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)
_RAGAS_IMPORT_LOCK = threading.Lock()
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _build_ragas_adapters(chat_model, embeddings):
    """Use LangChain's synchronous provider calls in worker threads for RAGAS."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    class ThreadedLLMWrapper(LangchainLLMWrapper):
        async def agenerate_text(
            self,
            prompt,
            n=1,
            temperature=0.01,
            stop=None,
            callbacks=None,
        ):
            loop = asyncio.get_running_loop()
            generate = super().generate_text
            return await loop.run_in_executor(
                None,
                functools.partial(
                    generate,
                    prompt,
                    n,
                    temperature,
                    stop,
                    callbacks,
                ),
            )

    class ThreadedEmbeddingsWrapper(LangchainEmbeddingsWrapper):
        async def aembed_query(self, text: str):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.embed_query, text)

        async def aembed_documents(self, texts: list[str]):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.embed_documents, texts)

    return ThreadedLLMWrapper(chat_model), ThreadedEmbeddingsWrapper(embeddings)


def _ensure_ragas_legacy_vertex_import() -> None:
    """Work around RAGAS 0.2's eager import of its unused legacy Vertex adapter."""
    module_name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        # This application evaluates with Azure OpenAI. RAGAS imports this
        # optional class unconditionally to build a provider type-check list.
        # A placeholder keeps the unused Vertex integration from blocking Azure.
        compatibility_module = types.ModuleType(module_name)
        compatibility_module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module_name] = compatibility_module


def _load_ragas_components():
    """Load async RAGAS metrics without patching the FastAPI event loop."""
    _ensure_ragas_legacy_vertex_import()
    import nest_asyncio

    # RAGAS 0.2.15 applies nest_asyncio globally while importing its executor.
    # This endpoint uses only async metric methods, so nested-loop patching is
    # unnecessary and interferes with FastAPI/AnyIO after the evaluation call.
    with _RAGAS_IMPORT_LOCK:
        original_apply = nest_asyncio.apply
        nest_asyncio.apply = lambda *_args, **_kwargs: None
        try:
            from ragas import SingleTurnSample
            from ragas.metrics import (
                Faithfulness,
                LLMContextPrecisionWithReference,
                LLMContextRecall,
                ResponseRelevancy,
            )
        finally:
            nest_asyncio.apply = original_apply
    return (
        SingleTurnSample,
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )


async def _run_ragas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score existing query outputs asynchronously with configured Azure models."""
    try:
        (
            SingleTurnSample,
            Faithfulness,
            ResponseRelevancy,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        ) = _load_ragas_components()

        evaluator_llm, evaluator_embeddings = _build_ragas_adapters(
            get_chat_model(), get_embeddings()
        )
        metrics = [
            Faithfulness(llm=evaluator_llm),
            ResponseRelevancy(
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
            ),
            LLMContextPrecisionWithReference(
                llm=evaluator_llm,
                name="context_precision",
            ),
            LLMContextRecall(llm=evaluator_llm),
        ]
        scores_by_record = []
        for row in rows:
            sample = SingleTurnSample(
                user_input=row["user_input"],
                response=row["response"],
                retrieved_contexts=row["retrieved_contexts"],
                reference=row["reference"],
            )
            scores = {}
            for metric in metrics:
                scores[metric.name] = _as_finite_float(
                    await metric.single_turn_ascore(sample)
                )
            scores_by_record.append(
                {name: scores.get(name) for name in METRIC_NAMES}
            )
        return scores_by_record
    except Exception as exc:
        logger.exception("RAGAS evaluation failed error_type=%s", type(exc).__name__)
        raise EvaluationServiceError("RAG evaluation could not be completed.") from exc


def _as_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class EvaluationService:
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        rows = []
        for record in request.records:
            response = await rag_service.query(
                QueryRequest(
                    query=record.question,
                    retrieval_strategy=request.retrieval_strategy,
                    prompt_strategy=request.prompt_strategy,
                    top_k=request.top_k,
                    candidate_k=request.candidate_k,
                    filters=request.filters,
                )
            )
            rows.append({
                "user_input": record.question,
                "reference": record.ground_truth,
                "response": response.answer,
                "retrieved_contexts": [
                    source.content for source in response.sources if source.content
                ],
            })

        scores = await _run_ragas(rows)
        if len(scores) != len(request.records):
            raise EvaluationServiceError("RAG evaluation returned incomplete results.")
        results = [
            EvaluationResult(question=record.question, **score)
            for record, score in zip(request.records, scores)
        ]
        averages = {}
        for name in METRIC_NAMES:
            values = [getattr(item, name) for item in results]
            usable = [value for value in values if value is not None]
            averages[name] = sum(usable) / len(usable) if usable else None
        logger.info("RAGAS evaluation completed records=%d", len(results))
        return EvaluationResponse(results=results, averages=averages)


evaluation_service = EvaluationService()
