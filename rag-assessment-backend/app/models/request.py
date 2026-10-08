from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import settings

MAX_CHAT_TURNS = 5


class QueryRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "query": "What is the standard family floater health insurance coverage?",
            "filters": {
                "category": "benefits",
                "subcategory": "insurance",
                "policy_id": "HR-BEN-2026",
            },
            "retrieval_strategy": "hybrid",
        }]
    })

    query: str = Field(..., min_length=1, max_length=settings.guardrail_max_query_chars)
    conversation_id: str | None = None
    chat_history: list[dict[str, str]] = Field(default_factory=list)

    @field_validator("chat_history")
    @classmethod
    def keep_latest_chat_turns(cls, value: list[dict[str, str]]) -> list[dict[str, str]]:
        # A normal turn contains one user message and one assistant reply.
        return value[-(MAX_CHAT_TURNS * 2):]

    top_k: int = Field(default=settings.rag_default_top_k, ge=1, le=50)
    candidate_k: int = Field(default=settings.rag_candidate_k, ge=1, le=100)

    use_hybrid: bool = True
    use_reranker: bool = True
    use_query_rewrite: bool = False
    use_hyde: bool = False
    use_multi_query: bool = False

    filters: dict[str, str] = Field(default_factory=dict)
    retrieval_strategy: str | None = None
    prompt_strategy: str = "zero_shot"
    reranker_strategy: str = "score"
    mmr_lambda: float = Field(default=settings.rag_mmr_lambda, ge=0.0, le=1.0)
    hybrid_vector_weight: float = Field(
        default=settings.rag_hybrid_vector_weight, gt=0.0, le=5.0
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query cannot be blank.")
        return value

    @field_validator("retrieval_strategy")
    @classmethod
    def validate_strategy(cls, value: str | None) -> str | None:
        if value is not None and value not in {"vector", "keyword", "hybrid"}:
            raise ValueError("retrieval_strategy must be vector, keyword, or hybrid.")
        return value

    @field_validator("prompt_strategy")
    @classmethod
    def validate_prompt_strategy(cls, value: str) -> str:
        if value not in {"zero_shot", "one_shot", "few_shot"}:
            raise ValueError("prompt_strategy must be zero_shot, one_shot, or few_shot.")
        return value

    @field_validator("reranker_strategy")
    @classmethod
    def validate_reranker_strategy(cls, value: str) -> str:
        if value not in {"score", "mmr"}:
            raise ValueError("reranker_strategy must be score or mmr.")
        return value

    @model_validator(mode="after")
    def candidates_cover_final_results(self) -> "QueryRequest":
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k.")
        return self


class ProcessRequest(BaseModel):
    document_id: str = Field(..., min_length=1)
    blob_name: str = Field(..., min_length=1)


class ChunkRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "document_id": "<document-id>",
            "text_blob_name": "extracted/<document-id>.json",
            "file_name": "03_Employee_Benefits_Insurance_Guide.pdf",
            "category": "benefits",
            "subcategory": "insurance",
            "policy_id": "HR-BEN-2026",
        }]
    })

    document_id: str = Field(..., min_length=1)
    text_blob_name: str = Field(..., min_length=1)
    file_name: str = ""
    category: str = Field(default="hr_policy", min_length=1)
    subcategory: str | None = None
    policy_id: str | None = None


class EmbedRequest(BaseModel):
    document_id: str = Field(..., min_length=1)
    chunk_blob_name: str = Field(..., min_length=1)


class EvaluationRecord(BaseModel):
    question: str = Field(..., min_length=1, max_length=settings.guardrail_max_query_chars)
    ground_truth: str = Field(..., min_length=1, max_length=10000)

    @field_validator("question", "ground_truth")
    @classmethod
    def values_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question and ground_truth must not be blank.")
        return value


class EvaluationRequest(BaseModel):
    records: list[EvaluationRecord] = Field(
        ..., min_length=1, max_length=settings.rag_evaluation_max_records
    )
    retrieval_strategy: str = "hybrid"
    prompt_strategy: str = "zero_shot"
    top_k: int = Field(default=settings.rag_default_top_k, ge=1, le=50)
    candidate_k: int = Field(default=settings.rag_candidate_k, ge=1, le=100)
    filters: dict[str, str] = Field(default_factory=dict)

    @field_validator("retrieval_strategy")
    @classmethod
    def validate_retrieval_strategy(cls, value: str) -> str:
        if value not in {"vector", "keyword", "hybrid"}:
            raise ValueError("retrieval_strategy must be vector, keyword, or hybrid.")
        return value

    @field_validator("prompt_strategy")
    @classmethod
    def validate_evaluation_prompt_strategy(cls, value: str) -> str:
        if value not in {"zero_shot", "one_shot", "few_shot"}:
            raise ValueError("prompt_strategy must be zero_shot, one_shot, or few_shot.")
        return value

    @model_validator(mode="after")
    def candidates_cover_final_results(self) -> "EvaluationRequest":
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k.")
        return self
