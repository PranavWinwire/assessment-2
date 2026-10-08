def extract_sources(documents: list[dict]) -> list[dict]:
    return [
        {
            "citation_id": f"SOURCE {index}",
            "document_id": doc.get("document_id"),
            "file_name": doc.get("file_name"),
            "category": doc.get("category"),
            "subcategory": doc.get("subcategory"),
            "policy_id": doc.get("policy_id"),
            "page": doc.get("page"),
            "slide": doc.get("slide"),
            "section": doc.get("section"),
            "chunk_id": doc.get("chunk_id"),
            "score": doc.get("score"),
            "content": doc.get("content"),
        }
        for index, doc in enumerate(documents, start=1)
    ]
