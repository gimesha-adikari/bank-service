# tests/test_liveness_backend.py
import os
import numpy as np
import pytest

from app.features.factory import build_liveness_backend

def _mk_plain(w=256, h=256, val=200):
    return np.full((h, w, 3), val, dtype=np.uint8)

def _mk_textured(w=256, h=256):
    rng = np.random.default_rng(42)
    img = (rng.integers(0, 255, size=(h, w, 3))).astype(np.uint8)
    # draw a few shapes for edges
    import cv2
    cv2.circle(img, (w//3, h//3), 40, (255, 255, 255), 2)
    cv2.line(img, (0,0), (w-1,h-1), (0,0,0), 2)
    return img

@pytest.mark.parametrize("backend", ["simple", "heuristic"])
def test_liveness_smoke(monkeypatch, backend):
    monkeypatch.setenv("APP_LIVENESS_BACKEND", backend)
    det = build_liveness_backend()
    if det is None:
        pytest.skip("Liveness backend is 'off'")
    # Both images should produce a score in [0,1]
    s1 = getattr(det, "score")( _mk_plain() )
    s2 = getattr(det, "score")( _mk_textured() )
    assert 0.0 <= float(s1) <= 1.0
    assert 0.0 <= float(s2) <= 1.0

@pytest.mark.skipif(os.getenv("CI") is None, reason="Keep heuristic expectations loose locally")
def test_heuristic_prefers_textured(monkeypatch):
    monkeypatch.setenv("APP_LIVENESS_BACKEND", "heuristic")
    det = build_liveness_backend()
    if det is None:
        pytest.skip("Liveness backend is 'off'")
    s_plain = float(det.score(_mk_plain()))
    s_tex   = float(det.score(_mk_textured()))
    assert s_tex >= s_plain  # textured should look more 'live' than flat plain patch
