from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings


def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_documents(documents):
    return get_splitter().split_documents(documents)
