import logging
import re

from app.core.config import settings
from app.core.exceptions import GuardrailBlockedError

logger = logging.getLogger(__name__)

# Natural employee phrasing that should remain in scope even when the configured
# term list contains only a different inflection or the canonical HR vocabulary.
HR_SCOPE_ALIASES = (
    r"\bleaves?\b",
    r"\btime[- ]off\b",
    r"\bdays?\s+off\b",
    r"\bsick\s+days?\b",
    r"\bpaid\s+time\s+off\b",
    r"\bparental\s+leave\b",
    r"\bmaternity\s+leave\b",
    r"\bpaternity\s+leave\b",
)


def screen_query(query: str) -> None:
    """Block common prompt attacks and questions outside the HR assistant scope."""
    if not settings.guardrails_enabled:
        return
    try:
        injection = re.compile(settings.guardrail_injection_patterns, re.IGNORECASE)
    except re.error:
        logger.exception("invalid configured injection pattern; request blocked")
        raise GuardrailBlockedError(
            "This request could not be processed safely.", reason="configuration_error"
        )
    if injection.search(query):
        logger.warning("guardrail blocked request category=prompt_injection action=block")
        raise GuardrailBlockedError(
            "This request was blocked because it conflicts with the assistant's safety rules.",
            reason="prompt_injection",
        )

    terms = [re.escape(term.strip()) for term in settings.guardrail_hr_terms.split(",")]
    in_scope = any(
        term and re.search(rf"(?<!\w){term}(?!\w)", query, re.IGNORECASE)
        for term in terms
    )
    if not in_scope:
        in_scope = any(re.search(pattern, query, re.IGNORECASE) for pattern in HR_SCOPE_ALIASES)
    if not in_scope:
        logger.info("guardrail blocked request category=out_of_scope action=refuse")
        raise GuardrailBlockedError(
            "I can help with questions about HR policies and employee handbooks.",
            reason="out_of_scope",
        )


def output_is_safe(answer: str) -> bool:
    """Detect output that appears to reveal hidden instructions."""
    try:
        pattern = re.compile(settings.guardrail_output_patterns, re.IGNORECASE)
    except re.error:
        logger.exception("invalid configured output pattern; output rejected")
        return False
    return pattern.search(answer) is None
