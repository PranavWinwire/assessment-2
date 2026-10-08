import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.exceptions import (
    DocumentIntelligenceError,
    DocumentNotFoundError,
    EvaluationServiceError,
    GuardrailBlockedError,
    InvalidDocumentError,
    ModelServiceError,
    SearchError,
    StorageError,
    UnsupportedFileTypeError,
)
from app.core.config import settings
from app.models.request import (
    ChunkRequest,
    EmbedRequest,
    EvaluationRequest,
    ProcessRequest,
    QueryRequest,
)
from app.models.response import EvaluationResponse, QueryResponse
from app.services.evaluation_service import evaluation_service
from app.services.ingestion_service import ingestion_service
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()


def _raise_http_error(exc: Exception) -> None:
    if isinstance(exc, UnsupportedFileTypeError):
        code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    elif isinstance(exc, InvalidDocumentError):
        code = status.HTTP_422_UNPROCESSABLE_CONTENT
    elif isinstance(exc, DocumentNotFoundError):
        code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, GuardrailBlockedError):
        code = status.HTTP_422_UNPROCESSABLE_CONTENT
    elif isinstance(
        exc,
        (
            StorageError,
            SearchError,
            ModelServiceError,
            DocumentIntelligenceError,
            EvaluationServiceError,
        ),
    ):
        code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        logger.exception("unexpected API failure")
        code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if isinstance(exc, GuardrailBlockedError):
        detail = {
            "code": "guardrail_blocked",
            "reason": exc.reason,
            "message": str(exc),
        }
    else:
        detail = str(exc) if code != 500 else "An internal error occurred."
    raise HTTPException(status_code=code, detail=detail)


@router.post("/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """Validate a source file and store it in Blob Storage."""
    try:
        logger.info("upload request received filename=%s", file.filename)
        data = await file.read(settings.max_upload_bytes + 1)
        return await ingestion_service.upload(
            file.filename or "", file.content_type or "application/octet-stream", data
        )
    except Exception as exc:
        _raise_http_error(exc)


@router.post("/documents/process")
async def process_document(request: ProcessRequest):
    try:
        logger.info("document processing started document_id=%s", request.document_id)
        return await ingestion_service.process(request.document_id, request.blob_name)
    except Exception as exc:
        _raise_http_error(exc)


@router.post("/documents/chunk")
async def chunk_document(request: ChunkRequest):
    try:
        logger.info("document chunking started document_id=%s", request.document_id)
        return await ingestion_service.chunk(
            request.document_id,
            request.text_blob_name,
            request.file_name,
            category=request.category,
            subcategory=request.subcategory,
            policy_id=request.policy_id,
        )
    except Exception as exc:
        _raise_http_error(exc)


@router.post("/documents/embed")
async def embed_document(request: EmbedRequest):
    try:
        logger.info("document indexing started document_id=%s", request.document_id)
        return await ingestion_service.embed(request.document_id, request.chunk_blob_name)
    except Exception as exc:
        _raise_http_error(exc)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        return await rag_service.query(request)
    except Exception as exc:
        _raise_http_error(exc)


@router.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
        return await rag_service.query(request, conversational=True)
    except Exception as exc:
        _raise_http_error(exc)


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    try:
        return await evaluation_service.evaluate(request)
    except Exception as exc:
        _raise_http_error(exc)
