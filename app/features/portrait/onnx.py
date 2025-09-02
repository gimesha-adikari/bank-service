# app/features/portrait/onnx.py
from __future__ import annotations
from typing import Tuple, Dict, Optional
import os, cv2, numpy as np

def _bytes_to_bgr(b: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(b, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _crop_with_margin(img: np.ndarray, x: int, y: int, w: int, h: int, margin: float) -> np.ndarray:
    H, W = img.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x0 = max(0, x - mx); y0 = max(0, y - my)
    x1 = min(W, x + w + mx); y1 = min(H, y + h + my)
    return img[y0:y1, x0:x1]

def _maybe_upscale(img: np.ndarray, min_side: int = 224) -> np.ndarray:
    h, w = img.shape[:2]
    m = min(h, w)
    if m >= min_side: return img
    s = float(min_side) / float(m)
    new = (int(round(w * s)), int(round(h * s)))
    return cv2.resize(img, new, interpolation=cv2.INTER_CUBIC)

class OnnxPortraitExtractor:
    def __init__(self) -> None:
        y0 = os.getenv("APP_YUNET_MODEL", "./models/face_detection_yunet.onnx")
        y1 = "./models/face/face_detection_yunet.onnx"
        yunet_path = y0 if os.path.isfile(y0) else (y1 if os.path.isfile(y1) else y0)
        self.det = cv2.FaceDetectorYN_create(yunet_path, "", (320, 320), 0.6, 0.3, 5000)
        self.min_size = int(float(os.getenv("APP_PORTRAIT_MIN_SIZE", os.getenv("APP_PORTRAIT_MIN", "64"))))
        self.margin   = float(os.getenv("APP_PORTRAIT_MARGIN", "0.35"))  # a bit wider by default
        self.min_out  = int(float(os.getenv("APP_PORTRAIT_MIN_OUT", "224")))

    def extract(self, front: bytes) -> Tuple[Optional[bytes], Dict]:
        img = _bytes_to_bgr(front)
        if img is None:
            return None, {"method": "yunet", "reason": "decode_failed"}

        H, W = img.shape[:2]
        self.det.setInputSize((W, H))
        num, faces = self.det.detect(img)
        if faces is None or len(faces) == 0:
            return None, {"method": "yunet", "reason": "no_faces"}

        faces = np.asarray(faces)
        idx = int(np.argmax(faces[:, 2] * faces[:, 3]))
        x, y, w, h = faces[idx][:4].astype(int)

        if w < self.min_size or h < self.min_size:
            return None, {"method": "yunet", "reason": "face_too_small", "bbox": [int(x), int(y), int(w), int(h)]}

        crop = _crop_with_margin(img, x, y, w, h, self.margin)
        crop = _maybe_upscale(crop, self.min_out)
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return None, {"method": "yunet", "reason": "encode_failed", "bbox": [int(x), int(y), int(w), int(h)]}
        return bytes(buf), {"method": "yunet", "bbox": [int(x), int(y), int(w), int(h)]}
