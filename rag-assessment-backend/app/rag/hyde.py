from app.rag.llm import invoke_chat
from app.rag.prompts import HYDE_PROMPT


async def generate_hypothetical_document(query: str) -> str:
    result = await invoke_chat(HYDE_PROMPT, {"query": query})
    return result.content.strip()
