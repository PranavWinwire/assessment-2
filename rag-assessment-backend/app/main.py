from fastapi import FastAPI
import logging
from uuid import uuid4

from app.api.routes import router
from app.core.config import settings
from app.core.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Backend API for a production RAG application",
)

app.include_router(router, prefix="/api/v1")


@app.middleware("http")
async def add_request_id(request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    logger.info(
        "request completed request_id=%s method=%s path=%s status=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}
