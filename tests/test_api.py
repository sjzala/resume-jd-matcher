from __future__ import annotations

from fastapi.testclient import TestClient

from matcher.api.main import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_match_empty_resume_returns_422() -> None:
    r = client.post("/match", json={"resume_text": "  ", "top_k": 3, "use_rerank": False})
    assert r.status_code == 422
