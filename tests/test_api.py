"""FastAPI integration tests using TestClient."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with (
        patch("medical_agent.rag.embedder.get_embedder"),
        patch("medical_agent.rag.reranker.get_reranker"),
        patch("medical_agent.agent.graph.get_compiled_graph", new_callable=AsyncMock),
    ):
        from medical_agent.api.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_eval_results_not_found(client, tmp_path):
    with patch("medical_agent.api.routes.eval.get_settings") as mock_s:
        mock_s.return_value.ragas_baseline_path = str(tmp_path / "nonexistent.json")
        resp = client.get("/eval/results")
        assert resp.status_code == 404


def test_eval_results_returns_data(client, tmp_path):
    baseline = {
        "naive": {
            "mode": "naive", "faithfulness": 0.61, "answer_relevancy": 0.58,
            "context_precision": 0.42, "context_recall": 0.55, "num_samples": 30,
        },
        "optimized": {
            "mode": "optimized", "faithfulness": 0.89, "answer_relevancy": 0.83,
            "context_precision": 0.76, "context_recall": 0.71, "num_samples": 30,
        },
    }
    baseline_path = tmp_path / "baseline_results.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline, f)

    with patch("medical_agent.api.routes.eval.get_settings") as mock_s:
        mock_s.return_value.ragas_baseline_path = str(baseline_path)
        resp = client.get("/eval/results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["naive"]["faithfulness"] == pytest.approx(0.61)
        assert data["optimized"]["faithfulness"] == pytest.approx(0.89)
