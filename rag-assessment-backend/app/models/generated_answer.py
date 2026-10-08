from pydantic import BaseModel, ConfigDict, Field, field_validator


class GeneratedAnswer(BaseModel):
    """Structured LLM response before citation labels are rendered."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        min_length=1,
        description="Employee-friendly answer supported only by the retrieved policy passages.",
    )
    citations: list[int] = Field(
        default_factory=list,
        description=(
            "1-based indexes of [SOURCE N] passages supporting factual answer claims. "
            "Must be non-empty for a factual answer; use an empty list only to abstain."
        ),
    )

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Answer must not be blank.")
        return value

    @field_validator("citations")
    @classmethod
    def citations_must_be_unique_and_positive(cls, value: list[int]) -> list[int]:
        if any(index < 1 for index in value):
            raise ValueError("Citation indexes start at 1.")
        if len(value) != len(set(value)):
            raise ValueError("Citation indexes must not repeat.")
        return value


class GroundednessVerdict(BaseModel):
    """Structured model-as-judge result for answer support by retrieved text."""

    model_config = ConfigDict(extra="forbid")

    is_grounded: bool
