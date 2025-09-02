# tests/test_debug_app.py
import numpy as np
from fastapi.testclient import TestClient

from app.debug_app import app

client = TestClient(app)

def _png_bytes(w=64, h=64, val=200):
    import cv2
    img = np.full((h, w, 3), val, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return bytes(buf)

def test_ping():
    r = client.get("/debug/ping")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

def test_liveness_endpoint_smoke(monkeypatch):
    # force simple liveness to avoid env surprises
    monkeypatch.setenv("APP_LIVENESS_BACKEND", "simple")
    files = {"image": ("a.png", _png_bytes(), "image/png")}
    r = client.post("/debug/liveness", files=files)
    assert r.status_code == 200
    data = r.json()
    assert "score" in data and 0.0 <= float(data["score"]) <= 1.0

def test_ocr_endpoint_smoke(monkeypatch):
    # keep 'simple' OCR by default for portability
    monkeypatch.setenv("APP_OCR_BACKEND", "simple")
    files = {"image": ("a.png", _png_bytes(), "image/png")}
    r = client.post("/debug/ocr?country=LK&doc=NIC", files=files)
    assert r.status_code == 200
    assert "raw_text" in r.json() or "text" in r.json()

def test_doc_classify_endpoint_smoke(monkeypatch):
    monkeypatch.setenv("APP_DOC_BACKEND", "heuristic")
    files = {"image": ("a.png", _png_bytes(), "image/png")}
    r = client.post("/debug/doc-classify", files=files)
    assert r.status_code == 200
    data = r.json()
    assert "class" in data and "confidence" in data
