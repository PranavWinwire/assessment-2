from app.models.generated_answer import GeneratedAnswer

FALLBACK_ANSWER = (
    "I do not have enough information in the available HR policy documents "
    "to answer that question."
)


def render_validated_answer(
    generated: GeneratedAnswer,
    source_count: int,
) -> str:
    """Check citation indexes and render API citation labels on the answer."""
    normalized = generated.answer.casefold()
    is_abstention = (
        "not have enough information" in normalized
        or "does not contain enough information" in normalized
    )
    if is_abstention:
        return generated.answer
    if not generated.citations:
        raise ValueError("An answer with factual content needs at least one citation.")
    if any(index > source_count for index in generated.citations):
        raise ValueError("The answer cites a source that was not retrieved.")
    labels = " ".join(f"[SOURCE {index}]" for index in generated.citations)
    return f"{generated.answer} {labels}"
