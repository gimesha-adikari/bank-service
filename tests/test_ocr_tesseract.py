# tests/test_ocr_tesseract.py
import os
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import pytesseract
    TESS_OK = bool(pytesseract.get_tesseract_version())
except Exception:
    TESS_OK = False

from app.features.ocr.tesseract import TesseractOcrEngine
from app.features.ocr.fields.nic import parse_nic_text


def _try_load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Try a few common fonts; fall back to PIL default
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\consola.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, 42)
        except Exception:
            pass
    return ImageFont.load_default()


@pytest.mark.skipif(not TESS_OK, reason="Tesseract binary not available on PATH")
def test_tesseract_text_simple(monkeypatch):
    # Bigger canvas + larger font for reliable OCR across environments
    W, H = 1200, 400
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    font = _try_load_font()
    text = "NIC: 912345678V\nName: TEST USER\nDOB: 1999-1-2"
    d.multiline_text((30, 40), text, fill=0, font=font, spacing=10)

    # Convert to BGR ndarray for engine
    rgb = img.convert("RGB")
    bgr = np.array(rgb)[:, :, ::-1].copy()

    # Slightly higher upscale for robustness in CI
    monkeypatch.setenv("APP_OCR_UPSCALE", "3.0")
    ocr = TesseractOcrEngine()
    raw = ocr.text(bgr)

    # Be robust to digit or letter misreads; check salient tokens exist
    assert "NIC" in raw
    assert "Name" in raw
    # At least a run of 5 digits should be present
    assert any(len(chunk) >= 5 for chunk in ["".join(c for c in raw if c.isdigit())])

    fields = ocr.extract_fields(bgr, country="LK", doc="NIC")
    assert "raw_text" in fields
    assert isinstance(fields["raw_text"], str)


def test_nic_parser_regex():
    txt = "ID Card\nNIC: 912345678V\nName: John A. Doe\nDOB: 1999/1/2"
    out = parse_nic_text(txt)
    assert out.get("nic") == "912345678V"
    assert out.get("nic_format") == "old_9X"
    assert out.get("name") == "John A. Doe"
    assert out.get("dob") == "1999-01-02"
