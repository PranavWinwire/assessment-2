from app.core.config import settings


def select_context_documents(documents: list[dict]) -> list[dict]:
    """Apply deterministic count and character budgets before prompt assembly."""
    selected: list[dict] = []
    remaining = settings.rag_context_max_chars
    for document in documents[: settings.rag_context_max_chunks]:
        if remaining <= 0:
            break
        content = str(document.get("content", ""))
        prefix = (
            f"[SOURCE {len(selected) + 1}]\n"
            f"File: {document.get('file_name', '')}\n"
            f"Page: {document.get('page')}\n"
            f"Section: {document.get('section', '')}\n"
            "Content: "
        )
        separator_cost = 2 if selected else 0
        available = remaining - len(prefix) - separator_cost
        if available <= 0:
            break
        content = content[:available]
        if not content.strip():
            continue
        selected.append({**document, "content": content})
        remaining -= len(prefix) + len(content) + separator_cost
    return selected


def build_context(documents: list[dict]) -> str:
    blocks = []

    for i, doc in enumerate(documents, start=1):
        blocks.append(
            f"[SOURCE {i}]\n"
            f"File: {doc.get('file_name', '')}\n"
            f"Page: {doc.get('page')}\n"
            f"Section: {doc.get('section', '')}\n"
            f"Content: {doc.get('content', '')}"
        )

    return "\n\n".join(blocks)
