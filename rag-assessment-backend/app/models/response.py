from pydantic import BaseModel, Field


class Source(BaseModel):
    citation_id: str | None = None
    document_id: str | None = None
    file_name: str | None = None
    category: str | None = None
    subcategory: str | None = None
    policy_id: str | None = None
    page: int | None = None
    slide: int | None = None
    section: str | None = None
    chunk_id: str | None = None
    score: float | None = None
    content: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    question: str
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


class EvaluationResponse(BaseModel):
    results: list[EvaluationResult]
    averages: dict[str, float | None]
