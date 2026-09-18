from __future__ import annotations

import json
import logging
import re
from typing import Optional, Dict, Any, List, Iterable, Tuple
from uuid import uuid4

from app.core.calib_log import write_row
from app.core.config import get_settings, segmented_threshold
from app.core.policy import decide
from app.features.factory import get_factory
from fastapi import APIRouter, Response, BackgroundTasks, HTTPException
from app.features.address.address_match import address_match_score

from .schemas import KycPayload, KycResult, AggregateResponse, CheckResult

try:
    import cv2
    import numpy as np
except Exception:
    cv2, np = None, None

router = APIRouter(prefix="/kyc", tags=["kyc"])
log = logging.getLogger("bank_ai.kyc")


# ---------------- small utils ----------------

def _log_after(event: str, payload: dict):
    try:
        log.info("KYC %s | %s", event, json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:
        pass


def _rotate(img_bytes: bytes, deg: int) -> bytes:
    if not cv2 or not np or (deg % 360 == 0):
        return img_bytes
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return img_bytes
        d = deg % 360
        if d == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif d == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif d == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return bytes(buf) if ok else img_bytes
    except Exception:
        return img_bytes


def _crop_box(img_bytes: bytes, x0: float, y0: float, x1: float, y1: float) -> Optional[bytes]:
    if not cv2 or not np:
        return None
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        xa, ya = int(max(0.0, min(1.0, x0)) * w), int(max(0.0, min(1.0, y0)) * h)
        xb, yb = int(max(0.0, min(1.0, x1)) * w), int(max(0.0, min(1.0, y1)) * h)
        if xb - xa <= 2 or yb - ya <= 2:
            return None
        crop = img[ya:yb, xa:xb]
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return bytes(buf) if ok else None
    except Exception:
        return None


def _heuristic_face_crops(front_b: bytes) -> List[bytes]:
    crops: List[bytes] = []
    rotations = [0, 90, 180, 270]
    boxes = [
        (0.03, 0.15, 0.43, 0.85),
        (0.57, 0.15, 0.97, 0.85),
        (0.20, 0.05, 0.80, 0.55),
        (0.20, 0.45, 0.80, 0.95),
        (0.32, 0.18, 0.68, 0.82),
    ]
    for deg in rotations:
        fb = _rotate(front_b, deg) if deg else front_b
        for (x0, y0, x1, y1) in boxes:
            c = _crop_box(fb, x0, y0, x1, y1)
            if c:
                crops.append(c)
    for deg in rotations:
        fb = _rotate(front_b, deg) if deg else front_b
        crops.append(fb)
    return crops


def _flatten_strings(obj: Any) -> Iterable[str]:
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, (int, float, bool)):
        yield str(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _flatten_strings(v)


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9,./\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------- address helpers (kept robust) ----------------

_STOP_WORDS = {
    "identity", "card", "national", "sri", "lanka", "nic", "number", "dob", "date",
    "name", "gender", "male", "female", "issue", "issued", "birth", "signature",
    "customer", "invoice", "bill", "account", "meter", "reading", "usage", "kwh",
}
_ADDR_CUES = r"road|rd|street|st|lane|ln|mawatha|mw|gama|pura|junction|jn|estate|watta|waththa|wattha|town|city|village"

def _keep_token(t: str) -> bool:
    if not t or len(t) < 2:
        return False
    if t.isdigit():
        return True
    if re.fullmatch(r"[a-z]{4,}", t) and not re.search(r"[aeiou]", t):
        return False
    if t in _STOP_WORDS:
        return False
    return True

def _tokset(s: str) -> set[str]:
    toks = re.split(r"[^a-z0-9]+", _norm(s))
    return {t for t in toks if _keep_token(t)}

def _numset(s: str) -> set[str]:
    return set(re.findall(r"\b\d+\b", _norm(s)))

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni

def _char_sim(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return float(SequenceMatcher(None, _norm(a), _norm(b)).ratio())

def _address_candidates(txt: str) -> List[str]:
    t = (txt or "").replace("\r", "\n")
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    cands: List[str] = []
    for ln in lines:
        ln_l = ln.lower()
        if any(sw in ln_l for sw in _STOP_WORDS):
            continue
        if ("," in ln) or re.search(_ADDR_CUES, ln_l):
            cands.append(ln)
    merged: List[str] = []
    i = 0
    while i < len(cands):
        cur = cands[i]
        j = i + 1
        while j < len(cands) and len(cands[j]) < 80:
            nxt = cands[j]
            if ("," in nxt) or re.search(_ADDR_CUES, nxt.lower()):
                cur = f"{cur}, {nxt}"
                j += 1
            else:
                break
        merged.append(cur)
        i = j
    if not merged:
        alpha_lines = [ln for ln in lines if sum(ch.isalpha() for ch in ln) >= 10]
        merged = sorted(alpha_lines, key=len, reverse=True)[:3]
    return merged[:6]

def _select_best_candidate(cands: List[str], ref: str) -> str:
    if not cands:
        return ""
    ref_toks = _tokset(ref)
    ref_nums = _numset(ref)
    best = cands[0]
    best_s = -1.0
    for c in cands:
        c_toks = _tokset(c)
        c_nums = _numset(c)
        j = _jaccard(c_toks, ref_toks)
        num_bonus = 0.15 if (c_nums and ref_nums and (c_nums & ref_nums)) else 0.0
        cs = _char_sim(c, ref) * 0.25
        s = j + num_bonus + cs
        if s > best_s:
            best_s, best = s, c
    return best

def _addr_from_bill(txt: str) -> str:
    m = re.search(r"(address|premises|service address|installation address)[^\n]*\n(.{10,200})", txt, flags=re.I)
    if m:
        block = m.group(2).splitlines()
        keep = [b.strip() for b in block if b.strip()]
        return ", ".join(keep[:3])[:240]
    cands = _address_candidates(txt)
    if cands:
        cands.sort(key=lambda s: (("," in s) + bool(re.search(_ADDR_CUES, s.lower())), len(s)), reverse=True)
        return cands[0][:240]
    return ""


# ---------------- OCR helpers ----------------

def _get_text_any_shape(det: Any, side: Optional[str] = None) -> str:
    if not isinstance(det, dict):
        return "".join(_flatten_strings(det) or [])
    for k in ("raw_text", "text"):
        v = det.get(k)
        if isinstance(v, str) and v.strip():
            return v
    if side:
        sub = det.get(side)
        if isinstance(sub, dict):
            for k in ("raw_text", "text"):
                v = sub.get(k)
                if isinstance(v, str) and v.strip():
                    return v
    for s in ("front", "back"):
        sub = det.get(s)
        if isinstance(sub, dict):
            for k in ("raw_text", "text"):
                v = sub.get(k)
                if isinstance(v, str) and v.strip():
                    return v
    return " ".join(list(_flatten_strings(det) or []))[:2000]


# Digit look-alike normalization for NIC regex
_DIGIT_FIX = str.maketrans({
    "O": "0", "o": "0", "D": "0",
    "I": "1", "l": "1", "|": "1", "!": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
})
def _fix_digit_lookalikes(s: str) -> str:
    return (s or "").translate(_DIGIT_FIX)


def _extract_nic_from_details(od: Any) -> Optional[str]:
    """Prefer structured fields; then regex over any text available."""
    if isinstance(od, dict):
        for k in ("nic", "id", "nic_number"):
            v = od.get(k)
            if isinstance(v, str) and v.strip():
                return re.sub(r"[^0-9A-Za-z]", "", v).upper()
        for side in ("front", "back"):
            sd = od.get(side)
            if isinstance(sd, dict):
                v = sd.get("nic") or sd.get("id")
                if isinstance(v, str) and v.strip():
                    return re.sub(r"[^0-9A-Za-z]", "", v).upper()

    whole = "".join(_flatten_strings(od) or [])
    norm = re.sub(r"[^0-9A-Za-z]", "", _fix_digit_lookalikes(whole)).upper()
    m = re.search(r"(\d{12})", norm) or re.search(r"(\d{9}[VX])", norm)
    return m.group(1) if m else None


# ---------------- routes ----------------

@router.get("/ping")
def ping():
    return {"ok": True, "feature": "kyc"}


@router.post("/face/match", response_model=KycResult)
def face_match(p: KycPayload):
    settings = get_settings()
    doc_front = p.docFront or p.docFrontImage
    if p.selfie is None or doc_front is None:
        raise HTTPException(status_code=400, detail="selfie and docFront/docFrontImage are required")

    score, details = get_factory().face_matcher().score(bytes(p.selfie), bytes(doc_front))
    details = dict(details or {})
    details["threshold"] = settings.face_threshold
    return KycResult(
        type="FACE_MATCH",
        score=float(score),
        passed=float(score) >= settings.face_threshold,
        detailsJson=json.dumps(details),
    )


@router.post("/liveness", response_model=KycResult)
def liveness(p: KycPayload):
    settings = get_settings()
    if p.selfie is None:
        raise HTTPException(status_code=400, detail="selfie is required")

    detector = get_factory().liveness()
    if detector is None:
        raise HTTPException(status_code=503, detail="liveness backend is disabled")
    score, details = detector.score(bytes(p.selfie))
    details = dict(details or {})
    details["threshold"] = settings.live_threshold
    return KycResult(
        type="LIVENESS",
        score=float(score),
        passed=float(score) >= settings.live_threshold,
        detailsJson=json.dumps(details),
    )


@router.post("/ocr/id", response_model=KycResult)
def ocr_id(p: KycPayload):
    settings = get_settings()
    front_b = p.docFrontImage or p.docFront
    back_b = p.docBackImage or p.docBack
    if front_b is None and back_b is None:
        raise HTTPException(status_code=400, detail="docFrontImage/docFront or docBackImage/docBack is required")

    score, details = get_factory().ocr().extract(
        bytes(front_b) if front_b else None,
        bytes(back_b) if back_b else None,
    )
    details = dict(details or {})
    details["threshold"] = settings.ocr_threshold
    return KycResult(
        type="OCR_ID",
        score=float(score),
        passed=float(score) >= settings.ocr_threshold,
        detailsJson=json.dumps(details),
    )


@router.post("/doc/class", response_model=KycResult)
def doc_class(p: KycPayload):
    settings = get_settings()
    front_b = p.docFrontImage or p.docFront
    back_b = p.docBackImage or p.docBack
    if front_b is None and back_b is None:
        raise HTTPException(status_code=400, detail="docFrontImage/docFront or docBackImage/docBack is required")

    score, details = get_factory().doc_classifier().classify(
        bytes(front_b) if front_b else None,
        bytes(back_b) if back_b else None,
    )
    details = dict(details or {})
    details["threshold"] = settings.doc_threshold
    return KycResult(
        type="DOC_CLASS",
        score=float(score),
        passed=float(score) >= settings.doc_threshold,
        detailsJson=json.dumps(details),
    )


@router.post("/aggregate", response_model=AggregateResponse)
def aggregate(p: KycPayload, response: Response, background_tasks: BackgroundTasks):
    settings = get_settings()
    fac = get_factory()

    request_id = str(uuid4())
    response.headers["X-Request-ID"] = request_id

    selfie_b   = bytes(p.selfie) if p.selfie is not None else None
    front_b    = bytes(p.docFrontImage or p.docFront) if (p.docFrontImage or p.docFront) else None
    back_b     = bytes(p.docBackImage or p.docBack) if (p.docBackImage or p.docBack) else None
    portrait_b = bytes(p.docPortraitImage) if p.docPortraitImage is not None else None
    bill_b     = bytes(p.addressProofImage) if p.addressProofImage is not None else None

    # -------- portrait finding (unchanged shape) --------
    portrait_source = "none"
    if portrait_b:
        portrait_source = "provided"
    elif front_b:
        extractor = fac.portrait()
        found = None
        if extractor is not None:
            for deg in (0, 90, 180, 270):
                fb = _rotate(front_b, deg) if deg else front_b
                try:
                    crop, _ = extractor.extract(fb)
                except Exception:
                    crop = extractor.extract(fb)
                if crop:
                    found = crop
                    portrait_source = "auto" if deg == 0 else f"auto_rot{deg}"
                    break
            if not found:
                for c in _heuristic_face_crops(front_b):
                    try:
                        crop, _ = extractor.extract(c)
                    except Exception:
                        crop = extractor.extract(c)
                    if crop:
                        found = crop
                        portrait_source = "auto_heuristic"
                        break
        portrait_b = found if found else front_b
        if not found:
            portrait_source = "front_fallback"

    # -------- face / liveness / ocr / docclass --------
    face_score = None
    face_details: Dict[str, Any] = {"source": portrait_source}
    matcher = fac.face_matcher()

    if selfie_b is not None and (portrait_b is not None or front_b is not None):
        doc_face_candidates: List[bytes] = [portrait_b] if portrait_b else []
        if portrait_source == "front_fallback" and front_b:
            doc_face_candidates.extend(_heuristic_face_crops(front_b))
        face_details["candidates"] = len(doc_face_candidates)

        best_s = -1.0
        for idx, cand in enumerate(doc_face_candidates[:12]):
            try:
                s, extra = matcher.score(selfie_b, cand)
                log.info("FACE extra debug: %s", extra)
            except TypeError:
                s, extra = matcher.score_pair(selfie_b, cand)
            if float(s) > best_s:
                best_s = float(s)
                face_details.update(extra or {})
                face_details["winning_candidate"] = 0 if idx == 0 else idx
        if best_s >= 0.0:
            face_score = best_s

    live_score = None
    live_details: Dict[str, Any] = {}
    if selfie_b is not None and fac.liveness() is not None:
        ls, lextra = fac.liveness().score(selfie_b)
        live_score = float(ls)
        live_details.update(lextra or {})

    ocr_score = None
    ocr_details: Dict[str, Any] = {}
    if front_b is not None or back_b is not None:
        os, odet = fac.ocr().extract(front_b, back_b)
        ocr_score = float(os)
        ocr_details.update(odet or {})

    doc_score = None
    doc_details: Dict[str, Any] = {}
    country = (p.meta or {}).get("countryHint")
    doc_class = (p.meta or {}).get("docClassHint")
    if front_b is not None or back_b is not None:
        ds, ddet = fac.doc_classifier().classify(front_b, back_b)
        doc_score = float(ds)
        doc_details.update(ddet or {})
        country = country or doc_details.get("country")
        doc_class = doc_class or doc_details.get("class")

    # -------- thresholds --------
    face_thr = segmented_threshold("APP_FACE_THRESHOLD", country, doc_class, settings.face_threshold)
    live_thr = segmented_threshold("APP_LIVE_THRESHOLD", country, doc_class, settings.live_threshold)
    ocr_thr  = segmented_threshold("APP_OCR_THRESHOLD",  country, doc_class, settings.ocr_threshold)
    doc_thr  = segmented_threshold("APP_DOC_THRESHOLD",  country, doc_class, settings.doc_threshold)
    addr_thr = segmented_threshold("APP_ADDRESS_THRESHOLD", country, doc_class, getattr(settings, "address_threshold", 0.75))
    valid_thr= segmented_threshold("APP_VALID_THRESHOLD",  country, doc_class, 1.0)

    checks: Dict[str, Dict[str, Any]] = {
        "FACE_MATCH": {"score": face_score, "passed": (face_score is not None and face_score >= face_thr), "threshold": face_thr, "details": face_details},
        "LIVENESS":   {"score": live_score, "passed": (live_score is not None and live_score >= live_thr), "threshold": live_thr, "details": live_details},
        "OCR_ID":     {"score": ocr_score,  "passed": (ocr_score  is not None and ocr_score  >= ocr_thr),  "threshold": ocr_thr,  "details": ocr_details},
        "DOC_CLASS":  {"score": doc_score,  "passed": (doc_score  is not None and doc_score  >= doc_thr),  "threshold": doc_thr,  "details": doc_details},
    }

    # -------- address match --------
    addr_score_final: float = 0.0
    addr_details_final: Dict[str, Any] = {}

    try:
        if back_b or bill_b:
            addr_score_lib, addr_dbg = address_match_score(fac.ocr(), back_b, bill_b)
            if addr_dbg and (addr_dbg.get("id_address") or addr_dbg.get("bill_address")):
                addr_score_final = float(addr_score_lib or 0.0)
                addr_details_final = {"source": "ocr_raw", **addr_dbg}
    except Exception as e:
        log.debug("ADDRESS_MATCH library path failed: %s", e)

    if not addr_details_final:
        id_back_text = _get_text_any_shape(ocr_details, side="back") if ocr_details else ""
        if not id_back_text and back_b:
            _s, det_b = fac.ocr().extract(None, back_b)
            id_back_text = _get_text_any_shape(det_b, side="back")

        bill_text = ""
        if bill_b:
            _s2, det_f = fac.ocr().extract(bill_b, None)
            bill_text = _get_text_any_shape(det_f, side="front")

        id_front_text = ""
        if not id_back_text and front_b:
            _s3, det_front = fac.ocr().extract(front_b, None)
            id_front_text = _get_text_any_shape(det_front, side="front")

        bill_addr = _addr_from_bill(bill_text)
        id_cands = _address_candidates(id_back_text)
        if id_front_text:
            id_cands += _address_candidates(id_front_text)
        id_addr = _select_best_candidate(id_cands, bill_addr) if bill_addr else (id_cands[0] if id_cands else "")

        A, B = _tokset(id_addr), _tokset(bill_addr)
        j = _jaccard(A, B)
        num_bonus = 0.15 if (_numset(id_addr) & _numset(bill_addr)) else 0.0
        cs = _char_sim(id_addr, bill_addr) * 0.25
        addr_score_final = min(1.0, j + num_bonus + cs) if (id_addr and bill_addr) else 0.0

        addr_details_final = {"source": "heuristic", "id_address": id_addr, "bill_address": bill_addr}

    checks["ADDRESS_MATCH"] = {
        "score": addr_score_final,
        "passed": addr_score_final >= addr_thr,
        "threshold": addr_thr,
        "details": addr_details_final,
    }

    try:
        log.info('ADDR DEBUG | id="%s" | bill="%s" | score=%.3f',
                 checks["ADDRESS_MATCH"]["details"].get("id_address", ""),
                 checks["ADDRESS_MATCH"]["details"].get("bill_address", ""),
                 float(checks["ADDRESS_MATCH"]["score"] or 0.0))
    except Exception:
        pass

    # -------- NIC validity (FIXED) --------
    nic = _extract_nic_from_details(ocr_details)

    # If NIC not found, rotate inputs and retry OCR quickly (handles vertical/sideways text)
    if not nic and (front_b or back_b):
        for deg in (90, 270, 180):
            f2 = _rotate(front_b, deg) if front_b else None
            b2 = _rotate(back_b, deg) if back_b else None
            try:
                _os2, od2 = fac.ocr().extract(f2, b2)
            except Exception:
                od2 = {}
            nic = _extract_nic_from_details(od2 or {})
            if nic:
                # Optionally keep a better raw_text for downstream inspection
                if isinstance(od2, dict):
                    raw2 = (od2.get("raw_text") or od2.get("text") or "")
                    if raw2 and isinstance(ocr_details, dict):
                        if len(raw2) > len(str(ocr_details.get("raw_text", ""))):
                            ocr_details["raw_text"] = raw2
                break

    valid_score = 1.0 if nic else 0.0
    checks["DOC_VALIDITY"] = {
        "score": valid_score, "passed": valid_score >= valid_thr, "threshold": valid_thr,
        "details": {"nic": nic}
    }

    # -------- decision + logs --------
    decision, reasons = decide(settings, checks)

    if settings.calibration_log:
        try:
            write_row(
                dirpath=settings.calibration_dir,
                request_id=request_id,
                instance_id=settings.instance_id,
                decision=decision,
                reasons=reasons,
                checks=checks,
                country=country,
                doc_class=doc_class,
            )
        except Exception:
            pass

    background_tasks.add_task(
        _log_after,
        "AGGREGATE",
        {
            "request_id": request_id,
            "decision": decision,
            "reasons": reasons,
            "country": country,
            "doc_class": doc_class,
            "checks": {k: {"score": v["score"], "passed": v["passed"], "threshold": v["threshold"], "source": v.get("details", {}).get("source")} for k, v in checks.items()},
        },
    )

    return AggregateResponse(
        decision=decision,
        reasons=reasons,
        checks=[
            CheckResult(
                type=k,
                score=v["score"],
                passed=v["passed"],
                details={**(v.get("details") or {}), "threshold": v.get("threshold")},
            )
            for k, v in checks.items()
        ],
    )
