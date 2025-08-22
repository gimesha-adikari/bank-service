import base64, shutil, pytest
from io import BytesIO
from PIL import Image, ImageDraw

tess = shutil.which("tesseract")
pytestmark = pytest.mark.skipif(tess is None, reason="tesseract not installed")

def _text_png_b64(lines, size=(600,200)):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    y = 20
    for ln in lines:
        d.text((20, y), ln, fill="black")
        y += 32
    buf = BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def test_ocr_id_extracts_fields(client):
    b64 = _text_png_b64([
        "Name: Test User",
        "DOB: 1999-01-01",
        "ID: ABC1234567"
    ])
    r = client.post("/api/v1/kyc/ocr/id", json={"docFrontImage": b64})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["type"] == "OCR_ID"
    assert data["score"] >= 0.75
    assert data["passed"] in (True, False)
    # ensure fields are present in detailsJson
    assert "ABC1234567" in data["detailsJson"]
    assert "1999-01-01" in data["detailsJson"]
    assert "Test User" in data["detailsJson"]

def test_ocr_id_missing_payload_400(client):
    r = client.post("/api/v1/kyc/ocr/id", json={"docFrontImage": None, "docBackImage": None})
    assert r.status_code == 400
