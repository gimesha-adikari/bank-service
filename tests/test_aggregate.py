import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter

def _png_b64(img):
    buf = BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _checker(size=(128,128), blur=False):
    img = Image.new("L", size, 0)
    d = ImageDraw.Draw(img)
    for y in range(0,size[1],8):
        for x in range(0,size[0],8):
            if ((x//8 + y//8) % 2)==0:
                d.rectangle((x,y,x+7,y+7), fill=255)
    if blur: img = img.filter(ImageFilter.GaussianBlur(3))
    return img.convert("RGB")

def _text_img(lines, size=(700,260)):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    y = 24
    for ln in lines:
        d.text((20, y), ln, fill="black")
        y += 34
    return img

def test_aggregate_approve_path(client):
    selfie = _checker()                         # sharp -> good liveness
    docfront = _checker()                       # similar -> decent face score
    text = _text_img([
        "REPUBLIC OF SRI LANKA",
        "NATIONAL IDENTITY CARD",
        "Department of Registration of Persons",
        "Name: Test User",
        "DOB: 1999-01-01",
        "ID: ABC1234567"
    ])
    payload = {
        "selfie": _png_b64(selfie),
        "docFrontImage": _png_b64(docfront),
    }
    r = client.post("/api/v1/kyc/ocr/id", json={"docFrontImage": _png_b64(text)})
    assert r.status_code == 200

    # Combine for aggregate
    payload["docFrontImage"] = _png_b64(text)   # so doc_class & ocr see the text image
    r = client.post("/api/v1/kyc/aggregate", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["decision"] in ("APPROVE","UNDER_REVIEW")  # typically APPROVE with defaults
    assert any(c["type"]=="FACE_MATCH" for c in data["checks"])
    assert any(c["type"]=="LIVENESS" for c in data["checks"])
    assert any(c["type"]=="OCR_ID" for c in data["checks"])
    assert any(c["type"]=="DOC_CLASS" for c in data["checks"])
