import base64
from io import BytesIO
from PIL import Image
import random

def _png_b64(color=(255, 0, 0), size=(64, 64)):
    """Generate a solid-color PNG and return base64 string."""
    img = Image.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def test_face_match_identical_images_pass(client):
    b64 = _png_b64((20, 120, 200))
    payload = {"selfie": b64, "docFront": b64}
    r = client.post("/api/v1/kyc/face/match", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["type"] == "FACE_MATCH"
    assert 0.95 <= data["score"] <= 1.0     # dHash should be very similar
    assert data["passed"] is True

def test_face_match_different_images_fail_or_low_score(client):
    b64_a = _png_b64((255, 0, 0))
    b64_b = _png_b64((0, 255, 0))
    r = client.post("/api/v1/kyc/face/match", json={"selfie": b64_a, "docFront": b64_b})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["type"] == "FACE_MATCH"
    assert 0.0 <= data["score"] <= 1.0
    # Expect below threshold used in your endpoint (0.85 in our step):
    assert data["passed"] in (False, None)
    assert data["score"] < 0.85

def test_face_match_missing_fields_400(client):
    r = client.post("/api/v1/kyc/face/match", json={"selfie": None, "docFront": None})
    assert r.status_code == 400
