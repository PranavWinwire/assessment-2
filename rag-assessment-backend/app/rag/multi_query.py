from app.rag.llm import invoke_chat
from app.rag.prompts import MULTI_QUERY_PROMPT


async def generate_queries(query: str) -> list[str]:
    result = await invoke_chat(MULTI_QUERY_PROMPT, {"query": query})

    return [line.strip() for line in result.content.splitlines() if line.strip()]
