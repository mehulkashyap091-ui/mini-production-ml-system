import os
import sys
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from serving.app import app, registry

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["known_users"] > 0

def test_warm_user_prediction(client):
    warm_user = next(iter(registry.store["user_features"].keys()))
    resp = client.post("/predict", json={"user_id": warm_user, "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "two_stage_ranker"
    assert body["is_cold_start"] is False
    assert body["fallback_used"] is False
    assert len(body["recommendations"]) == 5
    scores = [r["score"] for r in body["recommendations"]]
    assert scores == sorted(scores, reverse=True)

def test_cold_start_user_fallback(client):
    resp = client.post("/predict", json={"user_id": "UNSEEN_USER_999999", "top_k": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "cold_start_popularity"
    assert body["is_cold_start"] is True
    assert body["fallback_used"] is True
    assert len(body["recommendations"]) == 10

def test_exclude_items_respected(client):
    warm_user = next(iter(registry.store["user_features"].keys()))
    first_resp = client.post("/predict", json={"user_id": warm_user, "top_k": 5}).json()
    excluded = [first_resp["recommendations"][0]["stock_code"]]

    second_resp = client.post(
        "/predict", json={"user_id": warm_user, "top_k": 5, "exclude_items": excluded}
    )
    assert second_resp.status_code == 200
    returned_codes = [r["stock_code"] for r in second_resp.json()["recommendations"]]
    assert excluded[0] not in returned_codes

@pytest.mark.parametrize(
    "bad_payload",
    [
        {"user_id": "", "top_k": 5},
        {"user_id": "   ", "top_k": 5},
        {"user_id": "12345", "top_k": 0},
        {"user_id": "12345", "top_k": 100},
        {"top_k": 5},
        {"user_id": "12345", "top_k": "ten"},
    ],
)
def test_malformed_payload_rejected(client, bad_payload):
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422

def test_graceful_degradation_on_model_failure(client, monkeypatch):
    warm_user = next(iter(registry.store["user_features"].keys()))

    def broken_predict(*args, **kwargs):
        raise RuntimeError("Simulated LightGBM runtime failure")

    monkeypatch.setattr(registry.pipeline, "predict_proba", broken_predict)
    resp = client.post("/predict", json={"user_id": warm_user, "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "degraded_popularity_fallback"
    assert body["fallback_used"] is True
    assert len(body["recommendations"]) == 5
