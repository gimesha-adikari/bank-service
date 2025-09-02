from __future__ import annotations
import os
from typing import List, Sequence
import numpy as np

try:
    import onnxruntime as ort
except Exception as e:
    ort = None
    _IMPORT_ERR = e

def get_providers_from_env() -> List[str]:
    raw = os.getenv("APP_ONNX_PROVIDERS", "CPUExecutionProvider")
    return [p.strip() for p in raw.split(",") if p.strip()]

def new_session(model_path: str) -> "ort.InferenceSession":
    if ort is None:
        raise RuntimeError(f"onnxruntime not available: {_IMPORT_ERR}")
    providers = get_providers_from_env()
    opts = ort.SessionOptions()
    # conservative graph opts (safe on GPU/CPU)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)

def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = l2norm(a.reshape(1, -1))[0]
    b = l2norm(b.reshape(1, -1))[0]
    return float(np.dot(a, b))
