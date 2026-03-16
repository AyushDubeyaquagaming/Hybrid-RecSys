"""Tests for Redis connectivity reporting in the /health endpoint."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app
from app.services.model_service import ModelService


def _loaded_client(redis_enabled: bool, redis_reachable: bool):
    """Build a TestClient with a loaded model service and mocked Redis state."""
    svc = MagicMock(spec=ModelService)
    svc.is_loaded.return_value = True
    svc.n_users = 5
    svc.n_items = 10

    # Patch settings on the health router module
    settings_patch = MagicMock()
    settings_patch.REDIS_ENABLED = redis_enabled
    settings_patch.REDIS_HOST = "127.0.0.1"
    settings_patch.REDIS_PORT = 6379
    settings_patch.REDIS_DB = 0
    settings_patch.REDIS_PASSWORD = ""

    with patch("app.main.ModelService", return_value=svc):
        with patch("app.routes.health._settings", settings_patch):
            if redis_enabled:
                # Mock the _check_redis helper directly
                with patch(
                    "app.routes.health._check_redis", return_value=redis_reachable
                ):
                    with TestClient(app, raise_server_exceptions=True) as c:
                        yield c
            else:
                with TestClient(app, raise_server_exceptions=True) as c:
                    yield c


@pytest.fixture
def client_redis_enabled_up():
    yield from _loaded_client(redis_enabled=True, redis_reachable=True)


@pytest.fixture
def client_redis_enabled_down():
    yield from _loaded_client(redis_enabled=True, redis_reachable=False)


@pytest.fixture
def client_redis_disabled():
    yield from _loaded_client(redis_enabled=False, redis_reachable=False)


def test_health_redis_up_shows_connected(client_redis_enabled_up):
    resp = client_redis_enabled_up.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["redis_connected"] is True


def test_health_redis_down_still_200(client_redis_enabled_down):
    """Redis unreachable must NOT cause a 503 — health is still 200."""
    resp = client_redis_enabled_down.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["redis_connected"] is False


def test_health_redis_disabled_still_200(client_redis_disabled):
    """With REDIS_ENABLED=false, health is 200 and redis_connected key is absent."""
    resp = client_redis_disabled.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "redis_connected" not in data
