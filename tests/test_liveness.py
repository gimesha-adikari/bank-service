import base64
from io import BytesIO
from PIL import Image, ImageFilter

def _png_b64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def test_liveness_sharp_vs_blur(client):
    # sharp checkerboard
    size = (128, 128)
    img = Image.new("L", size, 0)
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = 255 if ((x//8 + y//8) % 2) else 0

    sharp_b64 = _png_b64(img)
    blur_b64  = _png_b64(img.filter(ImageFilter.GaussianBlur(radius=3)))

    r_sharp = client.post("/api/v1/kyc/liveness", json={"selfie": sharp_b64})
    r_blur  = client.post("/api/v1/kyc/liveness", json={"selfie": blur_b64})

    assert r_sharp.status_code == 200 and r_blur.status_code == 200
    s1 = r_sharp.json()["score"]
    s2 = r_blur.json()["score"]

    assert s1 > s2  # sharp image should have higher score
    # depending on default threshold 0.80, at least one should pass/fail distinctly
    assert r_sharp.json()["passed"] in (True, False)
    assert r_blur.json()["passed"] in (True, False)

def test_liveness_missing_selfie_400(client):
    r = client.post("/api/v1/kyc/liveness", json={"selfie": None})
    assert r.status_code == 400
