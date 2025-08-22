from io import BytesIO
from PIL import Image, ImageDraw
import base64
from _helpers import _png_b64, _face_like


def test_aggregate_approve_with_provided_portrait(client, monkeypatch):
    monkeypatch.setenv("APP_DOC_PORTRAIT_MODE", "provided")
    # ensure fresh settings for this request (in case cache behavior changes)
    from app.core.config import get_settings
    try:
        get_settings.cache_clear()
    except Exception:
        pass

    selfie = _face_like()
    portrait = selfie.copy()
    payload = {
        "selfie": _png_b64(selfie),
        "docPortraitImage": _png_b64(portrait),
    }
    r = client.post("/api/v1/kyc/aggregate", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["decision"] in ("APPROVE", "UNDER_REVIEW")
    face = next(c for c in data["checks"] if c["type"] == "FACE_MATCH")
    assert face["details"]["source"] in ("provided", "auto")
