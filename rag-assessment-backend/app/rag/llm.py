import logging
import asyncio
import time
from typing import Any, TypeVar

from langchain_openai import AzureChatOpenAI
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.core.exceptions import ModelServiceError

logger = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)


def get_chat_model():
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        azure_deployment=settings.azure_openai_chat_deployment,
        api_version=settings.azure_openai_api_version,
        temperature=0,
    )


def _is_transient_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429} or (
        isinstance(status_code, int) and status_code >= 500
    ):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__.casefold()
    return any(token in name for token in ("timeout", "connection", "ratelimit", "internalserver"))


async def _invoke_with_retry(runnable, values: dict[str, Any]):
    attempts = max(1, settings.rag_llm_retry_attempts)
    for attempt in range(1, attempts + 1):
        try:
            return await runnable.ainvoke(values)
        except (OutputParserException, ValidationError):
            raise
        except Exception as exc:
            if attempt >= attempts or not _is_transient_error(exc):
                raise
            delay = max(0.0, settings.rag_llm_retry_base_seconds) * (2 ** (attempt - 1))
            logger.warning(
                "transient language model failure retry=%d/%d delay_seconds=%.2f error_type=%s",
                attempt,
                attempts - 1,
                delay,
                type(exc).__name__,
            )
            await asyncio.sleep(delay)


async def invoke_chat(prompt, values: dict[str, Any]):
    """Invoke a prompt through Azure OpenAI and map provider failures safely."""
    started = time.perf_counter()
    try:
        result = await _invoke_with_retry(prompt | get_chat_model(), values)
        logger.info(
            "language model call completed latency_ms=%.0f",
            (time.perf_counter() - started) * 1000,
        )
        return result
    except Exception as exc:
        logger.exception("language model call failed")
        raise ModelServiceError("The language model request failed.") from exc


async def invoke_structured_chat(
    prompt,
    values: dict[str, Any],
    output_schema: type[ModelT],
) -> ModelT:
    """Invoke the chat deployment with a Pydantic-backed output schema."""
    started = time.perf_counter()
    try:
        model = get_chat_model().with_structured_output(
            output_schema,
            method="function_calling",
        )
        result = await _invoke_with_retry(prompt | model, values)
        logger.info(
            "structured model call completed latency_ms=%.0f",
            (time.perf_counter() - started) * 1000,
        )
        return result
    except (OutputParserException, ValidationError) as exc:
        raise OutputParserException("The model response did not match its schema.") from exc
    except Exception as exc:
        logger.exception("structured language model call failed")
        raise ModelServiceError("The language model request failed.") from exc
