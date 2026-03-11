"""
Tests for GET /health endpoint.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app
from app.services.model_service import ModelService


@pytest.fixture
def client_loaded():
    svc = MagicMock(spec=ModelService)
    svc.is_loaded.return_value = True
    svc.n_users = 42
    svc.n_items = 100
    with patch("app.main.ModelService", return_value=svc):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture
def client_not_loaded():
    svc = MagicMock(spec=ModelService)
    svc.is_loaded.return_value = False
    with patch("app.main.ModelService", return_value=svc):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def test_health_returns_200(client_loaded):
    resp = client_loaded.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_health_shows_dimensions(client_loaded):
    resp = client_loaded.get("/health")
    data = resp.json()
    assert data["n_users"] == 42
    assert data["n_items"] == 100


def test_health_unhealthy_when_not_loaded(client_not_loaded):
    resp = client_not_loaded.get("/health")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "unhealthy"
    assert data["model_loaded"] is False
