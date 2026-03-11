"""
Endpoint tests for POST /recommend.

These tests use a mocked ModelService so they don't require real artifacts.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app
from app.services.model_service import ModelService


KNOWN_USER = "user_001"
COLD_START_USER = "cold_start_user_xyz"

MOCK_RECS = [
    {
        "game_id": "1001",
        "game_name": "Aviator",
        "game_type": "crash",
        "provider": "Spribe",
        "score": 0.9,
        "rank": 1,
    },
    {
        "game_id": "1002",
        "game_name": "Slots",
        "game_type": "slots",
        "provider": "Pragmatic",
        "score": 0.8,
        "rank": 2,
    },
    {
        "game_id": "1003",
        "game_name": "Roulette",
        "game_type": "table",
        "provider": "Evolution",
        "score": 0.7,
        "rank": 3,
    },
]


MOCK_COLD_RECS = [
    {**r, "score": 0.0, "rank": i + 1}
    for i, r in enumerate(MOCK_RECS)
]


def make_mock_service(known_user=KNOWN_USER, recs=None, cold_start_recs=None):
    svc = MagicMock(spec=ModelService)
    svc.is_loaded.return_value = True
    svc.n_users = 100
    svc.n_items = 50

    if recs is None:
        recs = MOCK_RECS[:3]
    if cold_start_recs is None:
        cold_start_recs = MOCK_COLD_RECS[:3]

    def recommend(user_id, top_k, exclude_played):
        if user_id == known_user:
            return {
                "recommendations": recs[:top_k],
                "is_cold_start": False,
                "source": "lightfm",
            }
        else:
            return {
                "recommendations": cold_start_recs[:top_k],
                "is_cold_start": True,
                "source": "popularity_fallback",
            }

    svc.recommend.side_effect = recommend
    return svc


@pytest.fixture
def client():
    mock_svc = make_mock_service()
    with patch("app.main.ModelService", return_value=mock_svc):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def test_recommend_known_user(client):
    resp = client.post("/recommend", json={"user_id": KNOWN_USER, "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == KNOWN_USER
    assert len(data["recommendations"]) == 3
    assert data["metadata"]["is_cold_start"] is False
    assert data["metadata"]["source"] == "lightfm"
    # ranks should be 1..k
    ranks = [r["rank"] for r in data["recommendations"]]
    assert ranks == list(range(1, len(ranks) + 1))
    # scores should be present
    for rec in data["recommendations"]:
        assert "score" in rec
        assert "game_id" in rec


def test_recommend_cold_start_user(client):
    resp = client.post("/recommend", json={"user_id": COLD_START_USER})
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["is_cold_start"] is True
    assert data["metadata"]["source"] == "popularity_fallback"
    for rec in data["recommendations"]:
        assert rec["score"] == 0.0


def test_recommend_exclude_played(client):
    """Known items should be absent when exclude_played=true (mocked via known logic)."""
    resp = client.post(
        "/recommend",
        json={"user_id": KNOWN_USER, "top_k": 3, "exclude_played": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["excluded_played"] is True


def test_recommend_include_played(client):
    resp = client.post(
        "/recommend",
        json={"user_id": KNOWN_USER, "top_k": 3, "exclude_played": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["excluded_played"] is False


def test_recommend_custom_top_k(client):
    resp = client.post("/recommend", json={"user_id": KNOWN_USER, "top_k": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["top_k"] == 2
    assert len(data["recommendations"]) <= 2


def test_recommend_top_k_exceeds_max(client):
    resp = client.post("/recommend", json={"user_id": KNOWN_USER, "top_k": 99})
    assert resp.status_code == 422


def test_recommend_missing_user_id(client):
    resp = client.post("/recommend", json={"top_k": 5})
    assert resp.status_code == 422


def test_recommend_model_not_loaded():
    svc = MagicMock(spec=ModelService)
    svc.is_loaded.return_value = False
    with patch("app.main.ModelService", return_value=svc):
        with TestClient(app, raise_server_exceptions=True) as c:
            resp = c.post("/recommend", json={"user_id": KNOWN_USER})
    assert resp.status_code == 503
