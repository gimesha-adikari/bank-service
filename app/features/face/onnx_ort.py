# app/features/face/onnx_ort.py
from __future__ import annotations

import base64, os
from typing import Tuple, Dict, Any, Union
import cv2, numpy as np


def _to_bgr(x: Union[bytes, bytearray, str, np.ndarray]) -> np.ndarray:
    if isinstance(x, (bytes, bytearray)):
        arr = np.frombuffer(x, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("imdecode failed (bytes not an image?)")
        return img
    if isinstance(x, str):
        s = x.split(",", 1)[1] if x.startswith("data:") else x
        raw = base64.b64decode(s, validate=False)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("imdecode failed (b64 not an image?)")
        return img
    if isinstance(x, np.ndarray):
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if x.ndim == 2 else x
    raise TypeError(f"Unsupported image type: {type(x)}")


def _maybe_upscale(img: np.ndarray, target_min: int = 320) -> np.ndarray:
    h, w = img.shape[:2]
    m = min(h, w)
    if m >= target_min: return img
    s = float(target_min) / float(m)
    new = (int(round(w * s)), int(round(h * s)))
    return cv2.resize(img, new, interpolation=cv2.INTER_CUBIC)


def _largest_face_idx(faces: np.ndarray) -> int:
    areas = faces[:, 2] * faces[:, 3]
    return int(np.argmax(areas))


def _choose_model_path(env_key: str, candidates: list[str]) -> str:
    p = os.getenv(env_key)
    if p and os.path.isfile(p): return p
    for c in candidates:
        if os.path.isfile(c): return c
    return p or candidates[0]


class OnnxFaceMatcher:
    """
    YuNet + SFace via OpenCV contrib.
    score(selfie, doc) -> (score: float, extra: dict)
    """
    def __init__(self) -> None:
        yunet_path = _choose_model_path("APP_YUNET_MODEL",
                        ["./models/face_detection_yunet.onnx",
                         "./models/face/face_detection_yunet.onnx"])
        sface_path = _choose_model_path("APP_SFACE_MODEL",
                        ["./models/face_recognition_sface.onnx",
                         "./models/face/face_recognition_sface.onnx"])
        if not os.path.isfile(yunet_path): raise FileNotFoundError(f"YuNet not found: {yunet_path}")
        if not os.path.isfile(sface_path): raise FileNotFoundError(f"SFace not found: {sface_path}")

        try:
            self.detector = cv2.FaceDetectorYN_create(yunet_path, "", (320, 320), 0.6, 0.3, 5000)
        except AttributeError:
            raise RuntimeError("OpenCV build lacks FaceDetectorYN (needs opencv-contrib).")
        try:
            self.recognizer = cv2.FaceRecognizerSF_create(sface_path, "")
        except AttributeError:
            raise RuntimeError("OpenCV build lacks FaceRecognizerSF (needs opencv-contrib).")

        self.det_path, self.rec_path = yunet_path, sface_path
        self.threshold = float(os.getenv("APP_FACE_THRESHOLD", "0.36"))
        self.min_dim = int(float(os.getenv("APP_FACE_MIN_DIM", "320")))  # upsample small inputs

    def _detect_align_embed(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        img_bgr = _maybe_upscale(img_bgr, self.min_dim)
        h, w = img_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        num, faces = self.detector.detect(img_bgr)
        if faces is None or len(faces) == 0:
            raise ValueError("No face detected")
        faces = np.asarray(faces)
        idx = _largest_face_idx(faces)
        best_face = faces[idx]  # (x,y,w,h, l0x,l0y,..., l4x,l4y, score)
        face_aligned = self.recognizer.alignCrop(img_bgr, best_face)
        feature = self.recognizer.feature(face_aligned)
        extra = {
            "detected_faces": int(len(faces)),
            "chosen_index": int(idx),
            "bbox": {
                "x": float(best_face[0]), "y": float(best_face[1]),
                "w": float(best_face[2]), "h": float(best_face[3]),
            },
            "input_hw": [int(h), int(w)],
        }
        return feature, extra

    def score(self, selfie: Any, doc: Any) -> Tuple[float, Dict[str, Any]]:
        extra: Dict[str, Any] = {"backend": "onnx", "metric": "cosine", "threshold": self.threshold}
        try:
            img_selfie = _to_bgr(selfie)
            img_doc    = _to_bgr(doc)
        except Exception as e:
            extra["error"] = f"decode_error: {e}"
            return 0.0, extra

        try:
            f_selfie, ex_s = self._detect_align_embed(img_selfie)
            f_doc,    ex_d = self._detect_align_embed(img_doc)
            extra["selfie"], extra["doc"] = ex_s, ex_d
        except Exception as e:
            extra["error"] = f"detect_or_embed_error: {e}"
            return 0.0, extra

        try:
            cos = float(self.recognizer.match(f_selfie, f_doc, cv2.FaceRecognizerSF_FR_COSINE))
            l2  = float(self.recognizer.match(f_selfie, f_doc, cv2.FaceRecognizerSF_FR_NORM_L2))
        except Exception as e:
            extra["error"] = f"match_error: {e}"
            return 0.0, extra

        cos_norm = 0.5 * (cos + 1.0) if -1.0 <= cos <= 1.0 else cos
        extra.update({"cosine": cos, "cosine_norm": cos_norm, "l2": l2})
        return cos, extra
