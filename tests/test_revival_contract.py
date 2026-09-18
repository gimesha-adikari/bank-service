import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


FIXTURES = Path(__file__).resolve().parents[1] / "app" / "tests" / "fixtures"


@pytest.mark.parametrize(
    ("name", "expected"),
    (("approve.json", "APPROVE"), ("under_review.json", "UNDER_REVIEW"), ("reject.json", "REJECT")),
)
def test_preserved_aggregate_fixtures_keep_the_decision_contract(name: str, expected: str):
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    response = TestClient(app).post("/api/v1/kyc/aggregate", json=payload)

    assert response.status_code == 200, response.text
    assert response.headers.get("X-Request-ID")
    assert response.json()["decision"] == expected


def test_segmented_threshold_settings_are_read_from_environment(monkeypatch):
    monkeypatch.setenv("APP_FACE_THRESHOLD__LK__NIC", "0.91")
    from app.core.config import get_settings, segmented_threshold

    get_settings.cache_clear()
    assert segmented_threshold("APP_FACE_THRESHOLD", "LK", "NIC", 0.85) == 0.91
    get_settings.cache_clear()
