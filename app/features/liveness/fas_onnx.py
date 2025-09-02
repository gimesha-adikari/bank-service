# app/features/liveness/fas_onnx.py
from __future__ import annotations
import os
from typing import Dict, Tuple, Any

import cv2
import numpy as np
from app.features.common.onnxrt import new_session


def _to_bgr(img: Any) -> np.ndarray | None:
    if img is None:
        return None
    if isinstance(img, (bytes, bytearray)):
        arr = np.frombuffer(img, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if isinstance(img, np.ndarray):
        if img.ndim == 2:  return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:  return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.ndim == 3 and img.shape[2] == 3:  return img
        return None
    if isinstance(img, str) and os.path.exists(img):
        return cv2.imread(img, cv2.IMREAD_COLOR)
    if hasattr(img, "read"):
        try:
            data = img.read()
            if isinstance(data, (bytes, bytearray)):
                return _to_bgr(data)
        except Exception:
            return None
    return None


def _center_square_crop(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if h == w:
        return bgr
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return bgr[y0:y0 + s, x0:x0 + s]


def _preprocess_fas(bgr: np.ndarray, h: int, w: int) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        raise ValueError("Empty image for FAS preprocess.")
    bgr = _center_square_crop(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # NCHW
    x = np.expand_dims(x, 0)
    return x


class FasOnnx:
    """
    ONNX anti-spoof (RGB).
    Env:
      - APP_FAS_MODEL          : path to ONNX
      - APP_LIVENESS_THRESHOLD : default 0.5
      - APP_FAS_INPUT          : fallback input size if model isn't explicit
    """

    def __init__(self, model_path: str | None = None, input_size: int | None = None):
        model_path = model_path or os.getenv("APP_FAS_MODEL") or "./models/liveness/fas_rgb.onnx"
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"FAS model not found: {model_path}")

        self.session = new_session(model_path)
        self._in = self.session.get_inputs()[0].name
        self._out = self.session.get_outputs()[0].name

        # --- infer H×W from model shape (NCHW expected) ---
        shape = self.session.get_inputs()[0].shape  # e.g. [1, 3, 112, 112] or dynamic
        def _int(v):
            try: return int(v)
            except Exception: return None
        H = _int(shape[-2]) if isinstance(shape, (list, tuple)) and len(shape) >= 4 else None
        W = _int(shape[-1]) if isinstance(shape, (list, tuple)) and len(shape) >= 4 else None

        fallback = int(os.getenv("APP_FAS_INPUT", input_size or 112))
        self.input_h = H or fallback
        self.input_w = W or fallback

        self.threshold = float(os.getenv("APP_LIVENESS_THRESHOLD", "0.5"))

    def _infer_live_prob(self, x: np.ndarray) -> float:
        y = self.session.run([self._out], {self._in: x})[0]
        y = np.array(y).reshape(-1)
        if y.size == 1:
            return float(y[0])
        if np.any(y < 0) or np.any(y > 1):
            e = np.exp(y - np.max(y))
            p = e / (np.sum(e) + 1e-12)
        else:
            p = y / (np.sum(y) + 1e-12)
        return float(p[0])

    def predict(self, img: Any) -> Tuple[float, Dict]:
        bgr = _to_bgr(img)
        if bgr is None:
            raise ValueError("FasOnnx.predict: could not decode image to BGR.")
        x = _preprocess_fas(bgr, self.input_h, self.input_w)
        live_prob = self._infer_live_prob(x)
        passed = live_prob >= self.threshold
        return live_prob, {
            "backend": "fasonnx",
            "live_prob": live_prob,
            "passed": passed,
            "threshold": self.threshold,
            "input_hw": (self.input_h, self.input_w),
        }

    def score(self, img: Any) -> Tuple[float, Dict]:
        return self.predict(img)
