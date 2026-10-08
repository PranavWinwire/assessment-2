from langgraph.graph import END, START, StateGraph

from app.graph.nodes import generate, prepare_query, rerank, retrieve
from app.graph.state import RAGState


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("prepare_query", prepare_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)

    graph.add_edge(START, "prepare_query")
    graph.add_edge("prepare_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


rag_graph = build_graph()
