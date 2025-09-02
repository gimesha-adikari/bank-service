# app/features/face/arcface_onnx.py
from __future__ import annotations
import os
from typing import Dict, Tuple, Any

import cv2
import numpy as np

from app.features.common.onnxrt import new_session, l2norm, cosine_sim

_ARCFACE_H = 112
_ARCFACE_W = 112


def _to_bgr(img: Any) -> np.ndarray | None:
    """
    Accepts bytes, bytearray, numpy array (BGR/GRAY/BGRA), file path, or None.
    Returns BGR uint8 or None if cannot decode.
    """
    if img is None:
        return None

    # bytes / bytearray -> decode
    if isinstance(img, (bytes, bytearray)):
        arr = np.frombuffer(img, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return bgr

    # numpy array
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.ndim == 3 and img.shape[2] == 3:
            return img  # assume BGR
        return None

    # string path
    if isinstance(img, str) and os.path.exists(img):
        bgr = cv2.imread(img, cv2.IMREAD_COLOR)
        return bgr

    # file-like with .read()
    if hasattr(img, "read"):
        try:
            data = img.read()
            if isinstance(data, (bytes, bytearray)):
                return _to_bgr(data)
        except Exception:
            return None

    return None


def _preprocess_arcface(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> NCHW float32 in ArcFace scale (-1..1)."""
    if bgr is None or bgr.size == 0:
        raise ValueError("Empty image for ArcFace preprocess.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (_ARCFACE_W, _ARCFACE_H), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32)
    x = (x - 127.5) / 128.0            # ArcFace normalization
    x = np.transpose(x, (2, 0, 1))     # HWC -> CHW
    x = np.expand_dims(x, 0)           # NCHW
    return x


class ArcFaceMatcher:
    """
    Drop-in face matcher using ArcFace iResNet100 ONNX.
    - Accepts bytes / ndarray / path inputs.
    - Reads model path from APP_ARCFACE_MODEL (or ctor arg).
    - Uses ONNX Runtime providers from APP_ONNX_PROVIDERS.
    """

    def __init__(self, model_path: str | None = None):
        model_path = model_path or os.getenv("APP_ARCFACE_MODEL") or "./models/face/arcface_r100_glint360k.onnx"
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"ArcFace model not found: {model_path}")
        self.session = new_session(model_path)
        self._in_name = self.session.get_inputs()[0].name
        self._out_name = self.session.get_outputs()[0].name

    def embed(self, img: Any) -> np.ndarray:
        bgr = _to_bgr(img)
        if bgr is None:
            raise ValueError("ArcFaceMatcher.embed: could not decode image to BGR.")
        x = _preprocess_arcface(bgr)
        y = self.session.run([self._out_name], {self._in_name: x})[0]
        y = y.reshape(-1)
        y = l2norm(y)
        return y

    def score(self, selfie_img: Any, id_img: Any):
        e1 = self.embed(selfie_img)
        e2 = self.embed(id_img)
        sim = cosine_sim(e1, e2)  # raw cosine in [-1, 1]
        score01 = sim if 0.0 <= sim <= 1.0 else (sim + 1.0) / 2.0
        return score01, {
            "backend": "arcface",
            "embedding_dim": int(e1.shape[-1]),
            "similarity": float(sim),  # raw cosine for logging
            "score": float(score01)  # normalized used by checks
        }

