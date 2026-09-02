"""Authentication tests."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from approvavisa_engine.main import app
    return TestClient(app)


class TestAuth:
    def test_valid_key(self, client):
        response = client.get("/v1/specs", headers={"X-API-Key": "changeme"})
        assert response.status_code == 200

    def test_invalid_key(self, client):
        response = client.get("/v1/specs", headers={"X-API-Key": "invalid"})
        assert response.status_code == 403

    def test_missing_key(self, client):
        response = client.get("/v1/specs")
        assert response.status_code == 401

    def test_empty_key(self, client):
        response = client.get("/v1/specs", headers={"X-API-Key": ""})
        assert response.status_code == 401
