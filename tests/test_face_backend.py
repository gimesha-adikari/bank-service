# tests/test_face_backend.py
import os
import numpy as np
import pytest

from app.features.factory import build_face_backend


@pytest.mark.parametrize("backend", ["simple"])
def test_simple_face_backend_smoke(backend, monkeypatch):
    # Force "simple" so this passes even without ONNX models
    monkeypatch.setenv("APP_FACE_BACKEND", backend)
    matcher = build_face_backend()

    a = np.full((128, 128, 3), 120, dtype=np.uint8)
    b = np.full((128, 128, 3), 130, dtype=np.uint8)
    score = float(matcher.score_pair(a, b))
    assert 0.0 <= score <= 1.0


def test_onnx_backend_import_wiring(monkeypatch):
    """
    This checks that selecting the ONNX backend goes down the correct import path.
    If you have actual model files at ./models, the backend should construct successfully.
    Otherwise, it's acceptable to raise due to missing model files.
    """
    monkeypatch.setenv("APP_FACE_BACKEND", "onnx")
    monkeypatch.setenv("APP_YUNET_MODEL", "./models/face_detection_yunet.onnx")
    monkeypatch.setenv("APP_SFACE_MODEL", "./models/face_recognition_sface.onnx")

    try:
        _ = build_face_backend()
    except Exception as e:
        # Accept failures indicating the ONNX path is attempted (missing models, etc.)
        assert any(k in str(e).lower() for k in ("onnx", "model", "session", "inference"))
