from typing import Optional
from io import BytesIO
from PIL import Image
import numpy as np

def decode_image(b: Optional[bytes]) -> Optional[Image.Image]:
    if b is None:
        return None
    try:
        return Image.open(BytesIO(b)).convert("L")
    except Exception:
        return None

def dhash(img: Image.Image, hash_size: int = 8) -> np.ndarray:
    small = img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.uint8)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()

def dhash_similarity(img1: Image.Image, img2: Image.Image, hash_size: int = 8) -> float:
    h1 = dhash(img1, hash_size)
    h2 = dhash(img2, hash_size)
    dist = np.count_nonzero(h1 != h2)
    return 1.0 - (dist / h1.size)

def l1_similarity(img1: Image.Image, img2: Image.Image, size: tuple[int, int] = (64, 64)) -> float:
    i1 = img1.resize(size, Image.Resampling.LANCZOS)
    i2 = img2.resize(size, Image.Resampling.LANCZOS)
    a = np.asarray(i1, dtype=np.float32)
    b = np.asarray(i2, dtype=np.float32)
    mad = np.mean(np.abs(a - b)) / 255.0
    return float(1.0 - mad)

def perceptual_similarity(img1: Image.Image, img2: Image.Image, hash_size: int = 16, dhash_weight: float = 0.30) -> float:
    dh = dhash_similarity(img1, img2, hash_size=hash_size)
    l1 = l1_similarity(img1, img2)
    a = np.asarray(img1, dtype=np.float32); b = np.asarray(img2, dtype=np.float32)
    if a.std() < 2.0 and b.std() < 2.0:
        dhash_weight = min(dhash_weight, 0.20)
    return float(dhash_weight * dh + (1.0 - dhash_weight) * l1)
