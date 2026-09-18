"""
Feature Factory Module

This module provides a factory pattern implementation for creating and managing
various AI/ML feature backends used in the application. It handles face matching,
liveness detection, OCR, document classification and portrait extraction.

The factory supports multiple backend implementations for each feature and provides
fallback mechanisms if preferred backends fail to initialize.

Updated:
- Added optional GPU/ONNX backends:
  • Face matcher: ArcFace (app.features.face.arcface_onnx.ArcFaceMatcher)
  • Liveness: FAS ONNX (app.features.liveness.fas_onnx.FasOnnx)
  • OCR: PaddleOCR ONNX (app.features.ocr.paddleocr_onnx.PaddleOcrEngine)
- Kept all existing backends and fallbacks intact.
"""

from __future__ import annotations

import os
import logging
import inspect
import re
from typing import Any, Callable, Dict, Optional, Tuple

log = logging.getLogger(__name__)

try:
    from app.core.config import get_settings as _get_settings
except Exception:
    _get_settings = None

# --- existing simple/heuristic backends ---
from app.features.face.simple import SimpleFaceMatcher
from app.features.liveness.simple import SimpleLivenessDetector
from app.features.ocr.simple import SimpleOcrEngine
from app.features.doccls.simple import SimpleDocClassifier
from app.features.portrait.simple import SimplePortraitExtractor

# --- existing ONNX face matcher (OpenCV SFace/YuNet pipeline) ---
try:
    from app.features.face.onnx_ort import OnnxFaceMatcher
except Exception:
    OnnxFaceMatcher = None

# --- existing OCR (Tesseract) ---
try:
    from app.features.ocr.tesseract import TesseractOcrEngine
except Exception:
    TesseractOcrEngine = None

# --- existing liveness/doc heuristics ---
try:
    from app.features.liveness.heuristic import HeuristicLivenessDetector
except Exception:
    HeuristicLivenessDetector = None

try:
    from app.features.doccls.heuristic import HeuristicDocClassifier
except Exception:
    HeuristicDocClassifier = None

# --- optional ONNX portrait extractor (kept) ---
try:
    from app.features.portrait.onnx import OnnxPortraitExtractor
except Exception:
    OnnxPortraitExtractor = None

# --- NEW optional high-end backends (safe imports) ---
# Face: ArcFace ONNX matcher
try:
    from app.features.face.arcface_onnx import ArcFaceMatcher
except Exception:
    ArcFaceMatcher = None

# Liveness: FAS (anti-spoof) ONNX
try:
    from app.features.liveness.fas_onnx import FasOnnx
except Exception:
    FasOnnx = None

# OCR: PaddleOCR v3 ONNX (det+rec)
try:
    from app.features.ocr.paddleocr_onnx import PaddleOcrEngine
except Exception:
    PaddleOcrEngine = None

# Doc classification (optional advanced)
try:
    from app.features.doccls.layoutlm_onnx import LayoutLmDocClassifier
except Exception:
    LayoutLmDocClassifier = None


def _select_backend_name(settings: Any, key: str, default_value: str) -> str:
    env_val = os.getenv(f"APP_{key.upper()}")
    if env_val:
        return env_val.strip().lower()
    if settings is not None:
        val = getattr(settings, key, None)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return default_value


def _num(x: Any) -> Optional[float]:
    try:
        import numpy as _np
        _np_types = (int, float, bool, _np.floating, _np.integer)
    except Exception:
        _np_types = (int, float, bool)
    if isinstance(x, _np_types):
        return float(x)
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    return None


def _coerce_score(res: Any) -> float:
    score: Optional[float] = None
    if isinstance(res, (tuple, list)) and res:
        for v in res:
            score = _num(v)
            if score is not None:
                break
    elif isinstance(res, dict):
        for k in ("score", "similarity", "cosine", "sim", "match", "prob", "proba", "confidence"):
            if k in res:
                score = _num(res[k])
                if score is not None:
                    break
    else:
        score = _num(res)

    if score is None:
        raise TypeError(f"Unsupported score return type: {type(res)!r}")

    s = float(score)
    if -1.0 <= s <= 1.0:
        return s if 0.0 <= s <= 1.0 else (s + 1.0) / 2.0
    return max(0.0, min(1.0, s))


class _FaceScoreAdapter:
    def __init__(self, impl: Any, method_name: Optional[str] = None) -> None:
        self._impl = impl
        self._method = method_name  # e.g., 'score', 'compare', etc.

    def _call_backend(self, selfie_bgr, id_bgr):
        # If underlying backend exposes a scoring-like method, call it directly
        if self._method:
            m = getattr(self._impl, self._method)
            return m(selfie_bgr, id_bgr)

        # Else, fall back to embeddings → cosine
        if hasattr(self._impl, "embed") and callable(getattr(self._impl, "embed")):
            import numpy as np
            ea = np.asarray(self._impl.embed(selfie_bgr), dtype=np.float32).ravel()
            eb = np.asarray(self._impl.embed(id_bgr),  dtype=np.float32).ravel()
            denom = float(np.linalg.norm(ea) * np.linalg.norm(eb)) + 1e-8
            raw_cos = float(np.dot(ea, eb) / denom) if denom > 0.0 else 0.0
            # Return a tuple in the same shape the route expects
            return raw_cos, {"similarity": raw_cos, "algo": "cosine", "source": "adapter"}
        raise AttributeError("Face backend has neither a scoring method nor an 'embed' function.")

    @staticmethod
    def _to_tuple_normalized(out: Any) -> Tuple[float, Dict[str, Any]]:
        # Accept a variety of backend return shapes and normalize to (score[0..1], details)
        details: Dict[str, Any] = {}
        candidate = out
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            candidate = out[0]
            if len(out) >= 2 and isinstance(out[1], dict):
                details = dict(out[1])  # copy
        elif isinstance(out, dict):
            # Try to fish score-like values out of dicts
            for k in ("score", "similarity", "cosine", "sim", "match", "prob", "proba", "confidence"):
                if k in out:
                    candidate = out[k]
                    break
            details = dict(out)

        score = _coerce_score(candidate)
        if "normalized" not in details:
            details["normalized"] = True
        return float(score), details

    # Your route unpacks (score, extra), so both methods must return a tuple.
    def score(self, selfie_bgr, id_bgr) -> "_ScoreTuple":
        raw = self._call_backend(selfie_bgr, id_bgr)
        score, details = self._to_tuple_normalized(raw)
        return _ScoreTuple(score, details)

    def score_pair(self, selfie_bgr, id_bgr) -> "_ScoreTuple":
        return self.score(selfie_bgr, id_bgr)

    def __getattr__(self, item):
        return getattr(self._impl, item)




def _wrap_face_backend(obj: Any) -> Any:
    if hasattr(obj, "score_pair") and callable(getattr(obj, "score_pair")):
        return obj
    for name in ("score", "compare", "similarity", "match", "match_score", "predict"):
        if hasattr(obj, name) and callable(getattr(obj, name)):
            return _FaceScoreAdapter(obj, name)
    if hasattr(obj, "embed") and callable(getattr(obj, "embed")):
        return _FaceScoreAdapter(obj)
    return obj


class _ScoreTuple:
    __slots__ = ("score", "details")
    def __init__(self, score: float, details: Optional[dict] = None):
        self.score = float(score)
        self.details = details or {}
    def __float__(self) -> float:
        return self.score
    def __iter__(self):
        yield self.score
        yield self.details
    def __repr__(self) -> str:
        return f"_ScoreTuple(score={self.score:.3f}, details={self.details})"


class _LivenessScoreAdapter:
    def __init__(self, impl: Any, method_name: str = "score") -> None:
        self._impl = impl
        self._method = method_name if hasattr(impl, method_name) else None

    def score(self, bgr) -> _ScoreTuple:
        if self._method:
            val = getattr(self._impl, self._method)(bgr)
        elif hasattr(self._impl, "is_live"):
            val = 1.0 if bool(getattr(self._impl, "is_live")(bgr)) else 0.0
        else:
            raise AttributeError("Liveness backend has neither 'score' nor 'is_live'.")

        details: Dict[str, Any] = {}
        if isinstance(val, (tuple, list)) and len(val) >= 2 and isinstance(val[1], dict):
            details = val[1]

        score = _coerce_score(val)
        return _ScoreTuple(score, details)

    def __getattr__(self, item):
        return getattr(self._impl, item)


def _wrap_liveness_backend(obj: Any) -> Any:
    # Try common method names first; fall back to 'score'/'is_live'
    for name in ("score", "predict", "infer", "forward"):
        if hasattr(obj, name) and callable(getattr(obj, name)):
            return _LivenessScoreAdapter(obj, name)
    return _LivenessScoreAdapter(obj, "score")


class _OcrAdapter:

    def __init__(self, impl: Any) -> None:
        self._impl = impl

    def extract(self, front_b: Optional[bytes], back_b: Optional[bytes]) -> Tuple[float, Dict[str, Any]]:
        fields: Dict[str, Any] = {}
        score: float = 0.0
        used_backend = False

        if hasattr(self._impl, "extract"):
            out = getattr(self._impl, "extract")(front_b, back_b)
            if isinstance(out, (tuple, list)) and len(out) >= 2 and isinstance(out[1], dict):
                score = float(_coerce_score(out[0]))
                fields = dict(out[1])
                used_backend = True
            elif isinstance(out, dict):
                fields = dict(out)
                score = 1.0
                used_backend = True

        if not used_backend and hasattr(self._impl, "extract_fields"):
            bgr = None
            if front_b:
                bgr = self._bytes_to_bgr(front_b)
            elif back_b:
                bgr = self._bytes_to_bgr(back_b)
            if bgr is not None:
                try:
                    ef = getattr(self._impl, "extract_fields")(bgr, country="LK", doc="NIC")
                    if isinstance(ef, dict):
                        fields = dict(ef)
                        score = 1.0
                        used_backend = True
                except Exception:
                    pass

        raw_parts = []
        for img_b in (front_b, back_b):
            if img_b:
                txt = self._call_text_like(img_b)
                if txt:
                    raw_parts.append(txt)
        raw = "\n".join(raw_parts) if raw_parts else ""

        if raw:
            parsed = self._parse_common_id_fields(raw)
            for k in ("name", "dob", "id"):
                if k not in fields and k in parsed:
                    fields[k] = parsed[k]
            if "raw_text" not in fields:
                fields["raw_text"] = raw

        if not used_backend:
            score = 0.9 if raw else 0.0

        return score, fields

    def text(self, bgr) -> str:
        if hasattr(self._impl, "text") and callable(getattr(self._impl, "text")):
            return getattr(self._impl, "text")(bgr)
        ok, buf = self._bgr_to_png(bgr)
        if not ok:
            return ""
        return self._call_text_like(bytes(buf)) or ""

    def __getattr__(self, item):
        return getattr(self._impl, item)

    @staticmethod
    def _bytes_to_bgr(b: bytes):
        import numpy as np, cv2
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    @staticmethod
    def _bgr_to_png(bgr):
        import cv2
        return cv2.imencode(".png", bgr)

    def _call_text_like(self, img_bytes: bytes) -> str:
        bgr = self._bytes_to_bgr(img_bytes)
        for name in ("text", "ocr", "read", "predict", "infer", "recognize", "run"):
            if hasattr(self._impl, name) and callable(getattr(self._impl, name)):
                try:
                    res = getattr(self._impl, name)(bgr)
                except Exception:
                    try:
                        res = getattr(self._impl, name)(img_bytes)
                    except Exception:
                        continue
                if isinstance(res, bytes):
                    return res.decode(errors="ignore")
                if isinstance(res, str):
                    return res
                if isinstance(res, dict) and "raw_text" in res:
                    return str(res["raw_text"])
                return str(res)
        return ""

    @staticmethod
    def _parse_common_id_fields(txt: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]

        # --- NAME ---
        for ln in lines:
            m = re.search(r"(?i)\bname\b\s*[:\.\-]?\s*(.+)$", ln)
            if m:
                val = m.group(1).strip()
                val = re.split(r"(?i)\b(?:dob|id|address)\b\s*[:\.\-]?", val)[0].strip()
                if val:
                    out["name"] = val
                break

        # --- DOB (YYYY-MM-DD / YYYY/MM/DD) ---
        for ln in lines:
            m = re.search(r"(?i)\bdob\b\s*[:\.\-]?\s*([0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2})", ln)
            if m:
                out["dob"] = m.group(1)
                break

        # --- ID (alnum block) ---
        for ln in lines:
            m = re.search(r"(?i)\bid\b\s*[:\.\-]?\s*(.+)$", ln)
            if m:
                candidate = m.group(1).strip()
                candidate = re.split(r"(?i)\b(?:name|dob|address)\b\s*[:\.\-]?", candidate)[0].strip()
                id_norm = re.sub(r"[^A-Za-z0-9]", "", candidate)
                if id_norm:
                    out["id"] = id_norm
                break

        # --- ADDRESS (heuristic) ---
        # Strategy:
        # 1) If there's an "Address:" label, collect subsequent lines until a blank/next label.
        # 2) Else pick 2-4 consecutive lines that contain street cues / numbers.
        addr = None

        # 1) Labeled section
        for i, ln in enumerate(lines):
            if re.search(r"(?i)\baddress\b\s*[:\.\-]?", ln):
                piece = re.sub(r"(?i)\baddress\b\s*[:\.\-]?\s*", "", ln).strip()
                parts = [piece] if piece else []
                for j in range(i + 1, min(i + 6, len(lines))):
                    nxt = lines[j]
                    if re.search(r"(?i)\b(name|dob|id|sex|gender|nationality|issued|expiry)\b", nxt):
                        break
                    if not nxt.strip():
                        break
                    parts.append(nxt)
                joined = ", ".join([p for p in parts if p])
                if joined:
                    addr = joined
                break

        # 2) Heuristic window with street cues
        if not addr:
            def looks_like_addr(s: str) -> bool:
                cues = r"road|rd\.?|street|st\.?|lane|ln\.?|avenue|ave\.?|place|pl\.?|flat|apt|block|galle|kandy|colombo|no\."
                return bool(re.search(rf"(?i)\b({cues})\b", s)) or bool(re.search(r"\d{1,5}", s))

            windows = []
            for k in range(len(lines) - 1):
                window = ", ".join(lines[k:k + 3])
                if looks_like_addr(window):
                    windows.append(window)
            if windows:
                # choose the longest reasonable window
                windows = sorted(windows, key=lambda s: (-len(s), s))
                addr = windows[0]

        if addr:
            # light cleanup
            addr = re.sub(r"\s{2,}", " ", addr)
            addr = re.sub(r"\s*,\s*", ", ", addr).strip(" ,")
            out["address"] = addr

        return out


def _wrap_ocr_backend(obj: Any) -> Any:
    return _OcrAdapter(obj)


class FeatureFactory:
    """
    Factory class for creating and managing AI/ML feature backends.
    """

    def __init__(self) -> None:
        self.settings = _get_settings() if _get_settings is not None else None

        # --- FACE ---
        self._face_map: Dict[str, Callable[[], Any]] = {"simple": lambda: SimpleFaceMatcher()}
        if OnnxFaceMatcher is not None:
            self._face_map["onnx"] = lambda: OnnxFaceMatcher()
        if ArcFaceMatcher is not None:
            # Use env path inside ctor (ArcFaceMatcher reads APP_ARCFACE_MODEL); keep simple wrapper
            self._face_map["arcface"] = lambda: ArcFaceMatcher(os.getenv("APP_ARCFACE_MODEL"))

        # --- LIVENESS ---
        self._live_map: Dict[str, Callable[[], Any]] = {"simple": lambda: SimpleLivenessDetector()}
        if HeuristicLivenessDetector is not None:
            self._live_map["heuristic"] = lambda: HeuristicLivenessDetector()
        if FasOnnx is not None:
            self._live_map["fasonnx"] = lambda: FasOnnx(os.getenv("APP_FAS_MODEL"))

        # --- OCR ---
        self._ocr_map: Dict[str, Callable[[], Any]] = {"simple": lambda: SimpleOcrEngine()}
        if TesseractOcrEngine is not None:
            self._ocr_map["tesseract"] = lambda: TesseractOcrEngine()
        if PaddleOcrEngine is not None:
            self._ocr_map["paddleocr"] = lambda: PaddleOcrEngine(
                det_path=os.getenv("APP_OCR_DET_MODEL"),
                rec_path=os.getenv("APP_OCR_REC_MODEL"),
                dict_path=os.getenv("APP_OCR_DICT"),
                min_conf=float(os.getenv("APP_OCR_MIN_CONF", "0.7"))
            )

        # --- DOC CLASSIFIER ---
        self._doc_map: Dict[str, Callable[[], Any]] = {"simple": lambda: SimpleDocClassifier()}
        if HeuristicDocClassifier is not None:
            self._doc_map["heuristic"] = lambda: HeuristicDocClassifier()
        if LayoutLmDocClassifier is not None:
            self._doc_map["layoutlm"] = lambda: LayoutLmDocClassifier(os.getenv("APP_DOCCLASS_MODEL"))

        # --- PORTRAIT ---
        self._portrait_map: Dict[str, Callable[[], Any]] = {"simple": lambda: SimplePortraitExtractor()}
        if OnnxPortraitExtractor is not None:
            self._portrait_map["onnx"] = lambda: OnnxPortraitExtractor()

        # Cache
        self._face_obj: Optional[Any] = None
        self._live_obj: Optional[Any] = None
        self._ocr_obj: Optional[Any] = None
        self._doc_obj: Optional[Any] = None
        self._portrait_obj: Optional[Any] = None

    def _build(self, kind: str, mapping: Dict[str, Callable[[], Any]], fallback: str, wrap: Optional[Callable[[Any], Any]] = None):
        name = kind if kind in mapping else fallback
        ctor = mapping.get(name) or mapping.get(fallback)
        if ctor is None:
            raise RuntimeError(f"No constructor for backend '{kind}' and no fallback '{fallback}'.")
        try:
            obj = ctor()
        except Exception as e:
            strict_all  = os.getenv("APP_STRICT_BACKENDS", "").lower() in ("1","true","yes")
            is_face = mapping is getattr(self, "_face_map", None)
            strict_face = os.getenv("APP_STRICT_FACE", "").lower() in ("1","true","yes")
            if (strict_all or (is_face and strict_face)):
                raise
            if name != fallback and fallback in mapping:
                log.warning("Backend '%s' failed to build (%s). Falling back to '%s'.", name, e, fallback)
                obj = mapping[fallback]()
                name = fallback
            else:
                raise
        try:
            setattr(obj, "backend_name", name)
        except Exception:
            pass
        return wrap(obj) if wrap else obj

    def face_matcher(self) -> Any:
        if self._face_obj is None:
            kind = _select_backend_name(self.settings, "face_backend", "simple")
            self._face_obj = self._build(kind, self._face_map, "simple", wrap=_wrap_face_backend)
            try:
                actual = getattr(self._face_obj, "backend_name", kind if kind in self._face_map else "simple")
            except Exception:
                actual = kind if kind in self._face_map else "simple"
            log.info("Face backend: %s", actual)
        return self._face_obj

    def liveness(self) -> Optional[Any]:
        kind = _select_backend_name(self.settings, "liveness_backend", "simple")
        if kind == "off":
            return None
        if self._live_obj is None:
            self._live_obj = self._build(kind, self._live_map, "simple", wrap=_wrap_liveness_backend)
            log.info("Liveness backend: %s", kind if kind in self._live_map else "simple")
        return self._live_obj

    def ocr(self) -> Any:
        if self._ocr_obj is None:
            kind = _select_backend_name(self.settings, "ocr_backend", "simple")
            self._ocr_obj = self._build(kind, self._ocr_map, "simple", wrap=_wrap_ocr_backend)
            log.info("OCR backend: %s", kind if kind in self._ocr_map else "simple")
        return self._ocr_obj

    def doc_classifier(self) -> Any:
        if self._doc_obj is None:
            kind = _select_backend_name(self.settings, "doc_backend", "simple")
            self._doc_obj = self._build(kind, self._doc_map, "simple")
            log.info("Doc backend: %s", kind if kind in self._doc_map else "simple")
        return self._doc_obj

    def portrait(self) -> Optional[Any]:
        kind = _select_backend_name(self.settings, "portrait_backend", "simple")
        if kind == "off":
            return None
        if self._portrait_obj is None:
            self._portrait_obj = self._build(kind, self._portrait_map, "simple")
            log.info("Portrait backend: %s", kind if kind in self._portrait_map else "simple")
        return self._portrait_obj


_factory: Optional[FeatureFactory] = None
def get_factory() -> FeatureFactory:
    global _factory
    if _factory is None:
        _factory = FeatureFactory()
    return _factory

def build_face_backend():       return get_factory().face_matcher()
def build_liveness_backend():   return get_factory().liveness()
def build_ocr_backend():        return get_factory().ocr()
def build_doc_backend():        return get_factory().doc_classifier()
def build_portrait_backend():   return get_factory().portrait()

def reset_factory():
    global _factory
    _factory = None

__all__ = [
    "FeatureFactory", "get_factory", "reset_factory",
    "build_face_backend", "build_liveness_backend", "build_ocr_backend",
    "build_doc_backend", "build_portrait_backend",
]
