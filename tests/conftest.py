# tests/conftest.py
import os
import sys
from pathlib import Path
import pytest

# ---- Stable project root + working dir ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ---- Safe default envs for tests ----
os.environ.setdefault("APP_CALIBRATION_LOG", "0")
os.environ.setdefault("APP_CALIBRATION_DIR", str(PROJECT_ROOT / "calib_test"))
os.environ.setdefault("APP_INSTANCE_ID", "test")
os.environ.setdefault("APP_DOC_PORTRAIT_MODE", "off")
os.environ.setdefault("APP_FACE_THRESHOLD", "0.85")
os.environ.setdefault("APP_LIVE_THRESHOLD", "0.80")
os.environ.setdefault("APP_OCR_THRESHOLD",  "0.80")
os.environ.setdefault("APP_DOC_THRESHOLD",  "0.80")

# Import AFTER sys.path fix
from app.main import app  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

@pytest.fixture(scope="session")
def client():
    return TestClient(app)

# ---- NEW: clear settings cache per test so monkeypatch.setenv takes effect ----
@pytest.fixture(autouse=True)
def _reset_settings():
    # FastAPI handlers call get_settings() each request; it's cached via lru_cache.
    try:
        from app.core.config import get_settings
        get_settings.cache_clear()  # before test
        yield
        get_settings.cache_clear()  # after test (keep test isolation)
    except Exception:
        # If implementation changes and no cache exists, just continue.
        yield
