"""Tests for the /metrics Prometheus endpoint."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app
from app.services.model_service import ModelService


@pytest.fixture
def client():
    svc = MagicMock(spec=ModelService)
    svc.is_loaded.return_value = True
    svc.n_users = 10
    svc.n_items = 20
    with patch("app.main.ModelService", return_value=svc):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def test_metrics_endpoint_exists(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_metrics_endpoint_returns_prometheus_format(client):
    resp = client.get("/metrics")
    content_type = resp.headers.get("content-type", "")
    # Prometheus format uses text/plain with version header
    assert "text/plain" in content_type or "application/openmetrics-text" in content_type


def test_metrics_contains_http_requests_total(client):
    # Trigger a request so the counter is non-zero
    client.get("/health")
    resp = client.get("/metrics")
    assert "http_requests_total" in resp.text
