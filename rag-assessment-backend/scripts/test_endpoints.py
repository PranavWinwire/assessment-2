"""Exercise the running API's health and complete document ingestion flow."""

import argparse
import json
import mimetypes
import sys
from pathlib import Path
from typing import Any

import httpx


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = client.request(method, url, json=payload)
    if response.is_error:
        raise RuntimeError(
            f"{method} {url} failed ({response.status_code}): {response.text}"
        )
    return response.json()


def run(base_url: str, file_path: Path, query: str | None, timeout: float) -> None:
    """Call each endpoint in the upload -> process -> chunk -> embed flow."""
    base_url = base_url.rstrip("/")
    api_url = f"{base_url}/api/v1"
    with httpx.Client(timeout=timeout) as client:
        health = request_json(client, "GET", f"{base_url}/health")
        print(f"health: {health.get('status')}")

        media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        with file_path.open("rb") as document:
            response = client.post(
                f"{api_url}/documents/upload",
                files={"file": (file_path.name, document, media_type)},
            )
        if response.is_error:
            raise RuntimeError(
                f"POST /documents/upload failed ({response.status_code}): {response.text}"
            )
        uploaded = response.json()
        document_id = uploaded["document_id"]
        print(f"upload: {uploaded['status']} document_id={document_id}")

        processed = request_json(
            client,
            "POST",
            f"{api_url}/documents/process",
            {"document_id": document_id, "blob_name": uploaded["blob_name"]},
        )
        print(
            f"process: {processed['status']} characters={processed['character_count']}"
        )

        chunked = request_json(
            client,
            "POST",
            f"{api_url}/documents/chunk",
            {
                "document_id": document_id,
                "text_blob_name": processed["text_blob_name"],
                "file_name": uploaded["file_name"],
            },
        )
        print(f"chunk: {chunked['status']} chunks={chunked['chunk_count']}")

        indexed = request_json(
            client,
            "POST",
            f"{api_url}/documents/embed",
            {
                "document_id": document_id,
                "chunk_blob_name": chunked["chunk_blob_name"],
            },
        )
        print(f"embed: {indexed['status']} chunks={indexed['indexed_chunks']}")

        if query:
            answer = request_json(
                client,
                "POST",
                f"{api_url}/query",
                {"query": query, "retrieval_strategy": "hybrid"},
            )
            print("query response:")
            print(json.dumps(answer, indent=2, ensure_ascii=False))
        else:
            print("query: skipped (pass --query to test retrieval and generation)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="PDF, DOCX, TXT, PPTX, or CSV file",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--query", help="Optional question to ask after indexing")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    if not args.file.is_file():
        parser.error(f"file does not exist: {args.file}")
    try:
        run(args.base_url, args.file, args.query, args.timeout)
    except (httpx.HTTPError, RuntimeError, KeyError) as exc:
        print(f"Endpoint check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
