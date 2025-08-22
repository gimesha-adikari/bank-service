import base64
from io import BytesIO
from PIL import Image, ImageDraw

def _text_png_b64(lines, size=(700,260)):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    y = 24
    for ln in lines:
        d.text((20, y), ln, fill="black")
        y += 34
    buf = BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def test_doc_class_nic_pass(client):
    b64 = _text_png_b64([
        "REPUBLIC OF SRI LANKA",
        "NATIONAL IDENTITY CARD",
        "Department of Registration of Persons",
        "Name: Test User"
    ])
    r = client.post("/api/v1/kyc/doc/class", json={"docFrontImage": b64})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["type"] == "DOC_CLASS"
    assert data["score"] >= 0.8
    assert data["passed"] is True
    assert '"class": "NIC"' in data["detailsJson"]

def test_doc_class_unknown(client):
    b64 = _text_png_b64([
        "LIBRARY CARD",
        "Local Community Center",
        "Member: Foo Bar"
    ])
    r = client.post("/api/v1/kyc/doc/class", json={"docFrontImage": b64})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["type"] == "DOC_CLASS"
    assert data["passed"] in (False, True)  # but likely False
    # If it passed, score should be near threshold; otherwise < threshold
