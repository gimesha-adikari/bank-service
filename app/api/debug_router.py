# app/api/debug_router.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Any, Dict, Optional, List
import numpy as np
import cv2
import inspect

from app.features.factory import get_factory

router = APIRouter(prefix="/debug", tags=["debug"])


def _read_bytes(upload: UploadFile) -> bytes:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"Empty file: {upload.filename}")
    return data


def _to_bgr_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Unsupported image bytes")
    return img


def _to_bgr(upload: UploadFile) -> np.ndarray:
    data = _read_bytes(upload)
    return _to_bgr_from_bytes(data)


def _encode_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return bytes(buf)


@router.get("/ping")
def ping() -> Dict[str, str]:
    return {"status": "ok"}


@router.post("/face-score")
async def face_score(selfie: UploadFile = File(...), id: UploadFile = File(...)) -> JSONResponse:
    ff = get_factory()
    matcher = ff.face_matcher()
    s_bytes = _read_bytes(selfie)
    i_bytes = _read_bytes(id)
    details: Dict[str, Any] = {}
    try:
        score, details = matcher.score(s_bytes, i_bytes)  # prefer richer detail path when available
        score = float(score)
    except Exception:
        # Fall back to ndarray/score_pair path
        s_bgr = _to_bgr_from_bytes(s_bytes)
        i_bgr = _to_bgr_from_bytes(i_bytes)
        score = float(matcher.score_pair(s_bgr, i_bgr))
    info = {
        "score": score,
        "backend": getattr(matcher, "backend_name", details.get("algo", type(matcher).__name__)),
    }
    if isinstance(details, dict):
        info.update(details)
    # surface model paths if present (onnx)
    for k in ("det_path", "rec_path"):
        if hasattr(matcher, k):
            info[k] = getattr(matcher, k)
    return JSONResponse(info)


@router.post("/face-diag")
async def face_diag(selfie: UploadFile = File(...), id: UploadFile = File(...)) -> JSONResponse:
    # Alias to face-score; kept separate for future expansion
    return await face_score(selfie, id)


@router.post("/liveness")
async def liveness(image: UploadFile = File(...)) -> JSONResponse:
    ff = get_factory()
    det = ff.liveness()
    if det is None:
        raise HTTPException(status_code=503, detail="Liveness backend is OFF")
    bgr = _to_bgr(image)
    score = float(det.score(bgr))
    is_live = bool(getattr(det, "is_live")(bgr)) if hasattr(det, "is_live") else (score >= 0.80)
    return JSONResponse({"score": score, "is_live": is_live})


@router.post("/ocr")
async def ocr(image: UploadFile = File(...), country: str = "LK", doc: str = "NIC") -> JSONResponse:
    """
    Works with both 'tesseract' and 'simple' OCR:
      - If backend has extract_fields(), use it.
      - Else, try common single-image text methods and normalize to {'raw_text': str}.
    """
    ff = get_factory()
    ocr = ff.ocr()

    bgr = _to_bgr(image)

    if hasattr(ocr, "extract_fields"):
        try:
            out = ocr.extract_fields(bgr, country=country, doc=doc)  # type: ignore[attr-defined]
            if isinstance(out, dict):
                return JSONResponse(out)
        except Exception:
            pass

    # Try common text method names
    candidates: List[str] = ["text", "ocr", "read", "predict", "infer", "recognize", "run"]
    for name in candidates:
        if hasattr(ocr, name) and callable(getattr(ocr, name)):
            res = getattr(ocr, name)(bgr)
            if isinstance(res, bytes):
                return JSONResponse({"raw_text": res.decode(errors="ignore")})
            if isinstance(res, str):
                return JSONResponse({"raw_text": res})
            if isinstance(res, dict) and "raw_text" in res:
                return JSONResponse(res)
            return JSONResponse({"raw_text": str(res)})
    return JSONResponse({"raw_text": ""})


@router.post("/doc-classify")
async def doc_classify(
    image: UploadFile = File(...),
    back: Optional[UploadFile] = File(None),
) -> JSONResponse:
    """
    Supports both 1-image (heuristic) and 2-image (simple) classifiers.
    If a 2-image classifier is detected and 'back' is not provided, a blank
    back PNG is synthesized to keep this endpoint ergonomic.
    """
    ff = get_factory()
    clf = ff.doc_classifier()

    # Read front bytes (keep both bytes and decoded for flexibility)
    f_bytes = _read_bytes(image)
    f_bgr = _to_bgr_from_bytes(f_bytes)

    # Choose classify() or fallback to predict()
    if hasattr(clf, "classify"):
        m = getattr(clf, "classify")
        try:
            sig = inspect.signature(m)
        except (TypeError, ValueError):
            sig = None

        # Count user-facing positional params (exclude 'self')
        user_params = 0
        if sig:
            params = [
                p for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            user_params = len([p for p in params if p.name != "self"])

        # Two-input classifier: use provided back or synthesize blank, pass BYTES first (fallback to arrays)
        if user_params >= 2:
            if back is not None:
                b_bytes = _read_bytes(back)
                # decode just to get dims if needed later
                b_bgr = _to_bgr_from_bytes(b_bytes)
            else:
                h, w = f_bgr.shape[:2]
                blank = np.full((h, w, 3), 255, dtype=np.uint8)
                b_bgr = blank
                b_bytes = _encode_png_bytes(blank)

            # Try bytes-first (matches SimpleDocClassifier expectations)
            try:
                out = m(f_bytes, b_bytes)
            except Exception:
                out = m(f_bgr, b_bgr)
        else:
            # One-input classifier (heuristic). Try bytes first (in case another backend uses bytes),
            # then fallback to ndarray.
            try:
                out = m(f_bytes)
            except Exception:
                out = m(f_bgr)

        # Normalize output
        if isinstance(out, dict):
            klass = str(out.get("class", "UNKNOWN"))
            conf = float(out.get("confidence", 0.0))
            details: Any = out.get("details", {})
            return JSONResponse({"class": klass, "confidence": conf, "details": details})
        return JSONResponse({"class": str(out), "confidence": 0.0, "details": {}})

    # Fallback: predict()
    if hasattr(clf, "predict"):
        if back is not None:
            b_bytes = _read_bytes(back)
            try:
                label = clf.predict(f_bytes, b_bytes)  # type: ignore[arg-type]
            except Exception:
                b_bgr = _to_bgr_from_bytes(b_bytes)
                label = clf.predict(f_bgr, b_bgr)
        else:
            try:
                label = clf.predict(f_bytes)  # type: ignore[arg-type]
            except Exception:
                label = clf.predict(f_bgr)
        return JSONResponse({"class": str(label), "confidence": 0.0, "details": {}})

    raise HTTPException(status_code=500, detail="No suitable document classifier method found")
