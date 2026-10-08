"""Compare Azure AI Search retrieval modes for the same question."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.integrations.azure_search import AzureSearchClient


async def compare(question: str, top_k: int) -> dict:
    search = AzureSearchClient()
    results = {}
    for strategy in ("vector", "keyword", "hybrid"):
        documents = await search.search(
            query=question,
            top_k=top_k,
            strategy=strategy,
        )
        results[strategy] = [
            {
                "rank": rank,
                "score": document.get("score"),
                "file_name": document.get("file_name"),
                "document_id": document.get("document_id"),
                "chunk_id": document.get("chunk_id"),
                "page": document.get("page"),
                "slide": document.get("slide"),
                "section": document.get("section"),
                "content": document.get("content"),
            }
            for rank, document in enumerate(documents, start=1)
        ]
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 50:
        parser.error("--top-k must be between 1 and 50")
    print(json.dumps(asyncio.run(compare(args.query, args.top_k)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
