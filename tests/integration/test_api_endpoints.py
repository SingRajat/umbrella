import io
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.storage.chroma import storage_client

client = TestClient(app)


def test_api_health():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["chromadb"] == "connected"
    assert "version" in data


def test_api_document_lifecycle():
    content = b"Umbrella API test document for FastAPI verification."
    # 1. Upload document
    upload_res = client.post(
        "/api/v1/documents",
        files={"file": ("api_test.txt", io.BytesIO(content), "text/plain")},
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    doc_id = upload_data["doc_id"]
    assert upload_data["filename"] == "api_test.txt"
    assert upload_data["chunk_count"] >= 1

    try:
        # 2. List documents
        list_res = client.get("/api/v1/documents?page=1&page_size=10")
        assert list_res.status_code == 200
        list_data = list_res.json()
        assert any(d["doc_id"] == doc_id for d in list_data["documents"])
        assert list_data["pagination"]["total"] >= 1

        # 3. Get document detail
        detail_res = client.get(f"/api/v1/documents/{doc_id}")
        assert detail_res.status_code == 200
        detail_data = detail_res.json()
        assert detail_data["doc_id"] == doc_id
        assert detail_data["filename"] == "api_test.txt"

        # 4. Duplicate upload triggers 409 Conflict
        dup_res = client.post(
            "/api/v1/documents",
            files={"file": ("api_test_dup.txt", io.BytesIO(content), "text/plain")},
        )
        assert dup_res.status_code == 409

    finally:
        # 5. Delete document
        del_res = client.delete(f"/api/v1/documents/{doc_id}")
        assert del_res.status_code == 200
        del_data = del_res.json()
        assert del_data["status"] == "deleted"


def test_api_query_refusal_on_empty():
    res = client.post(
        "/api/v1/query",
        json={"query": "Random query without matching context", "doc_id": "non-existent"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "refused"
    assert "reason" in data


def test_api_eval_run():
    res = client.post("/api/v1/eval/run", json={})
    assert res.status_code == 200
    data = res.json()
    assert "run_id" in data
    assert "config_hash" in data
