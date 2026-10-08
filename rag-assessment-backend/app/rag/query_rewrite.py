from app.rag.llm import invoke_chat
from app.rag.prompts import QUERY_REWRITE_PROMPT


async def rewrite_query(query: str, history: list[dict[str, str]]) -> str:
    result = await invoke_chat(
        QUERY_REWRITE_PROMPT, {"query": query, "history": history}
    )
    return result.content.strip()
