from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_unsupported_extension_is_rejected_before_storage():
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("policy.xlsx", b"data", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_mismatched_mime_type_is_rejected():
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("policy.pdf", b"%PDF-1.7 content", "text/plain")},
    )
    assert response.status_code == 415


def test_non_text_mime_type_is_rejected_for_txt():
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("policy.txt", b"valid text", "application/pdf")},
    )
    assert response.status_code == 415


def test_non_text_mime_type_is_rejected_for_txt():
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("policy.txt", b"valid text", "application/pdf")},
    )
    assert response.status_code == 415


def test_empty_supported_file_is_rejected():
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("policy.txt", b"", "text/plain")},
    )
    assert response.status_code == 422


def test_blank_query_is_rejected():
    response = client.post("/api/v1/query", json={"query": "   "})
    assert response.status_code == 422


def test_out_of_scope_query_is_rejected_with_safe_message():
    response = client.post("/api/v1/query", json={"query": "What is the capital of France?"})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "guardrail_blocked"
    assert response.json()["detail"]["reason"] == "out_of_scope"
    assert "HR policies" in response.json()["detail"]["message"]


def test_evaluation_request_requires_labeled_records():
    response = client.post("/api/v1/evaluate", json={"records": []})
    assert response.status_code == 422


def test_evaluation_endpoint_returns_scored_results(monkeypatch):
    from app.api import routes
    from app.models.response import EvaluationResponse, EvaluationResult

    async def fake_evaluate(_request):
        return EvaluationResponse(
            results=[EvaluationResult(
                question="How do I request leave?",
                faithfulness=0.9,
                answer_relevancy=0.8,
                context_precision=0.7,
                context_recall=0.6,
            )],
            averages={
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "context_precision": 0.7,
                "context_recall": 0.6,
            },
        )

    monkeypatch.setattr(routes.evaluation_service, "evaluate", fake_evaluate)
    response = client.post("/api/v1/evaluate", json={
        "records": [{
            "question": "How do I request leave?",
            "ground_truth": "Submit a leave request.",
        }]
    })
    assert response.status_code == 200
    assert response.json()["results"][0]["faithfulness"] == 0.9
    assert response.json()["averages"]["context_recall"] == 0.6
