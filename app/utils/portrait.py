from typing import Optional, Tuple, Dict
from io import BytesIO
from PIL import Image
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

def _bytes_to_rgb(b: bytes) -> Optional[Image.Image]:
    try:
        return Image.open(BytesIO(b)).convert("RGB")
    except Exception:
        return None

def _crop_with_margin(img: Image.Image, box, margin: float) -> Image.Image:
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    x0 = max(0, x - mx); y0 = max(0, y - my)
    x1 = min(img.width, x + w + mx); y1 = min(img.height, y + h + my)
    return img.crop((x0, y0, x1, y1))

def detect_portrait(front_bytes: bytes, min_size: int = 64, margin: float = 0.2) -> Tuple[Optional[bytes], Dict]:
    img = _bytes_to_rgb(front_bytes)
    if img is None:
        return None, {"method": "none", "reason": "decode_failed"}

    if not _HAS_CV2:
        return None, {"method": "none", "reason": "cv2_missing"}

    arr = np.array(img)[:, :, ::-1]
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:
        return None, {"method": "haar", "reason": "cascade_load_failed"}

    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))
    if len(faces) == 0:
        return None, {"method": "haar", "reason": "no_faces"}

    x, y, w, h = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
    crop = _crop_with_margin(img, (x, y, w, h), margin)
    buf = BytesIO(); crop.save(buf, format="PNG")
    return buf.getvalue(), {"method": "haar", "bbox": [int(x), int(y), int(w), int(h)]}
