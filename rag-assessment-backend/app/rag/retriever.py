from app.integrations.azure_search import AzureSearchClient


class RAGRetriever:
    def __init__(self):
        self.search = AzureSearchClient()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        filters: dict | None = None,
        hybrid: bool = True,
        strategy: str | None = None,
        vector_weight: float = 1.0,
    ):
        return await self.search.search(
            query=query,
            top_k=candidate_k,
            filters=filters or {},
            hybrid=hybrid,
            strategy=strategy,
            vector_weight=vector_weight,
        )
