from typing import Tuple
from PIL import Image, ImageFilter
import numpy as np

LAPLACIAN_KERNEL = np.array([[0, 1, 0],
                             [1,-4, 1],
                             [0, 1, 0]], dtype=np.float32)

def _to_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)

def _conv2_gray(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(gray, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.empty_like(gray)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * kernel)
    return out

def focus_measure(img: Image.Image) -> float:
    g = _to_gray(img)
    lap = _conv2_gray(g, LAPLACIAN_KERNEL)
    return float(lap.var())

def color_variance(img: Image.Image) -> float:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return float(arr.reshape(-1, 3).var(axis=0).mean())

def liveness_score(img: Image.Image) -> Tuple[float, dict]:
    f = focus_measure(img)
    c = color_variance(img)

    # crude normalization; tune later
    f_norm = min(f / 1500.0, 1.0)
    c_norm = min(c / 1500.0, 1.0)

    score = 0.6 * f_norm + 0.4 * c_norm
    return float(score), {"lap_var": f, "color_var": c}
