"""API integration tests using HTTPX TestClient."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from approvavisa_engine.main import app
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "changeme"}


class TestHealthEndpoint:
    def test_health_check(self, client):
        """Health endpoint should not require auth."""
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["engine"] == "approvavisa-engine"

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "engine" in response.json()


class TestAuthEndpoints:
    def test_specs_without_key(self, client):
        """Should reject requests without API key."""
        response = client.get("/v1/specs")
        assert response.status_code == 401

    def test_specs_with_invalid_key(self, client):
        """Should reject invalid API key."""
        response = client.get("/v1/specs", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 403

    def test_specs_with_valid_key(self, client, auth_headers):
        """Should accept valid API key."""
        response = client.get("/v1/specs", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 40

    def test_specs_search(self, client, auth_headers):
        """Should search specs by query."""
        response = client.get("/v1/specs?q=canada", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert any(s["code"] == "CA" for s in data)

    def test_spec_by_code(self, client, auth_headers):
        """Should get spec by country code."""
        response = client.get("/v1/specs/US", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "US"
        assert data["name"] == "United States"

    def test_spec_not_found(self, client, auth_headers):
        """Should 404 for unknown country."""
        response = client.get("/v1/specs/ZZZ", headers=auth_headers)
        assert response.status_code == 404


class TestValidateEndpoint:
    def test_validate_missing_image(self, client, auth_headers):
        """Should reject validate requests without image."""
        response = client.post(
            "/v1/validate",
            json={"country_code": "US"},
            headers=auth_headers,
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_validate_invalid_country(self, client, auth_headers):
        """Should 404 for unknown country in validate."""
        response = client.post(
            "/v1/validate",
            json={"image": "dGVzdA==", "country_code": "ZZZ"},
            headers=auth_headers,
        )
        assert response.status_code == 404
