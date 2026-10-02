from uuid import uuid4
from fastapi.testclient import TestClient
from app.main import app
from pydantic import ValidationError


def test_health_is_public(client):
    response = client.get("/health", headers={"X-Bank-Core-Auth": ""})

    assert response.status_code == 200


def test_kyc_ping_requires_internal_credential(client):
    response = TestClient(app).get("/api/v1/kyc/ping")

    assert response.status_code == 401
    assert "secret" not in response.text.lower()


def test_kyc_ping_rejects_wrong_internal_credential(client):
    response = client.get("/api/v1/kyc/ping", headers={"X-Bank-Core-Auth": "wrong"})

    assert response.status_code == 401


def test_kyc_ping_accepts_internal_credential(client):
    response = client.get("/api/v1/kyc/ping")

    assert response.status_code == 200


def test_every_kyc_route_requires_internal_credential():
    client = TestClient(app)
    routes = [
        ("GET", "/api/v1/kyc/ping"),
        ("POST", "/api/v1/kyc/face/match"),
        ("POST", "/api/v1/kyc/liveness"),
        ("POST", "/api/v1/kyc/ocr/id"),
        ("POST", "/api/v1/kyc/doc/class"),
        ("POST", "/api/v1/kyc/aggregate"),
    ]

    for method, path in routes:
        response = client.request(method, path, json={} if method == "POST" else None)
        assert response.status_code == 401


def test_aggregate_requires_trusted_bank_user_id(client):
    response = client.post("/api/v1/kyc/aggregate", json={})

    assert response.status_code == 400


def test_aggregate_accepts_trusted_bank_user_id(client):
    response = client.post(
        "/api/v1/kyc/aggregate",
        json={"bankUserId": str(uuid4())},
    )

    assert response.status_code == 200


def test_service_secret_uses_exact_external_environment_name(monkeypatch):
    from app.core.config import Settings

    monkeypatch.delenv("BANK_SERVICE_AUTH_SECRET", raising=False)
    monkeypatch.setenv("APP_BANK_SERVICE_AUTH_SECRET", "test-service-secret-012345678901234567890123")

    try:
        Settings(_env_file=None)
    except ValidationError:
        return
    raise AssertionError("APP-prefixed alias must not satisfy BANK_SERVICE_AUTH_SECRET")


def test_service_secret_rejects_short_and_placeholder_values(monkeypatch):
    from app.core.config import Settings

    for value in ("short", "CHANGE_ME_TO_A_RANDOM_SECRET", " CHANGE_ME_TO_A_RANDOM_SECRET "):
        monkeypatch.setenv("BANK_SERVICE_AUTH_SECRET", value)
        try:
            Settings(_env_file=None)
        except ValueError:
            continue
        except ValidationError:
            continue
        raise AssertionError("invalid service credential was accepted")
