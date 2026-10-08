from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "HR Policy Assistant API"
    debug: bool = Field(default=False, validation_alias="APP_DEBUG")

    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_chat_deployment: str = "gpt-4.1-mini"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    azure_openai_api_version: str = "2024-10-21"
    azure_embedding_dimensions: int = 1536

    azure_search_endpoint: str = ""
    azure_search_api_key: str = ""
    azure_search_index: str = "rag-index"

    azure_storage_connection_string: str = ""
    azure_storage_container: str = "documents"

    document_intelligence_endpoint: str = ""
    document_intelligence_key: str = ""
    document_intelligence_model: str = "prebuilt-layout"

    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 150
    rag_default_top_k: int = 5
    rag_candidate_k: int = 20
    rag_score_threshold: float = 0.0
    rag_low_confidence_vector_threshold: float = Field(default=0.5, ge=0.0)
    rag_low_confidence_keyword_threshold: float = Field(default=3.0, ge=0.0)
    rag_low_confidence_hybrid_threshold: float = Field(default=0.02, ge=0.0)
    rag_hybrid_vector_weight: float = Field(default=1.0, gt=0.0, le=5.0)
    rag_context_max_chunks: int = Field(default=10, ge=1, le=50)
    rag_context_max_chars: int = Field(default=20000, ge=100, le=200000)
    rag_mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    rag_llm_retry_attempts: int = Field(default=3, ge=1, le=10)
    rag_llm_retry_base_seconds: float = Field(default=0.5, ge=0.0, le=60.0)
    rag_groundedness_check_enabled: bool = True
    guardrails_enabled: bool = True
    guardrail_max_query_chars: int = Field(default=2000, ge=1, le=20000)
    guardrail_injection_patterns: str = (
        r"ignore (all |any |the )?(previous|prior|above) instructions|"
        r"disregard (all |any |the )?(previous|prior|above) instructions|"
        r"reveal (the )?(system|developer) prompt|"
        r"show (me )?(the )?(system|developer) prompt|"
        r"enter developer mode|jailbreak"
    )
    guardrail_hr_terms: str = (
        "hr,human resources,employee,employees,leave,vacation,sick,benefit,benefits,"
        "payroll,salary,compensation,policy,handbook,holiday,holidays,pto,termination,"
        "maternity,paternity,unpaid,manager,workplace,conduct,insurance,retirement"
    )
    guardrail_output_patterns: str = (
        r"(system|developer) prompt|internal instructions|hidden instructions"
    )
    rag_evaluation_max_records: int = Field(default=20, ge=1, le=100)
    max_upload_bytes: int = 20 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
