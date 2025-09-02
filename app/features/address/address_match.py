# app/features/address_match.py
from __future__ import annotations
import re
from typing import Tuple, Dict, Any, Optional

def _normalize(s: str) -> str:
    s = (s or "")
    try:
        s = s.encode("ascii", "ignore").decode()
    except Exception:
        pass
    s = s.lower()
    s = re.sub(r"[^a-z0-9/,\-\s]", " ", s)
    s = re.sub(r"\b(sri\s*lanka|lk)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_address_text(raw: str, kind: str) -> str:
    # kind: "id" or "bill"
    raw = (raw or "").replace("\r", "\n")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    joined = " ".join(lines)

    cues_id = ["place of residence", "residence", "address"]
    cues_bill = ["service address", "billing address", "installation address", "address"]
    cues = cues_id if kind == "id" else cues_bill

    idx = -1
    low = joined.lower()
    for cue in cues:
        i = low.find(cue)
        if i != -1:
            idx = i + len(cue)
            break

    if idx != -1:
        window = joined[idx: idx + 160]
    else:
        # Fallback: longest plausible address-like lines
        cand = [ln for ln in lines if re.search(
            r"\d+|road|rd|street|st|lane|ln|mawatha|gama|pura|town|city|village", ln.lower())]
        cand.sort(key=len, reverse=True)
        window = " ".join(cand[:3])[:200]

    # Stop at next likely label
    window = re.split(r"(?i)\b(name|dob|id|nic|account|meter|date|bill|invoice|customer)\b", window)[0]
    return window.strip()

def _token_set(s: str) -> set[str]:
    toks = re.split(r"[^a-z0-9]+", _normalize(s))
    return {t for t in toks if len(t) >= 2 and not t.isdigit()}

def _num_set(s: str) -> set[str]:
    return set(re.findall(r"\b\d+\b", _normalize(s)))

def address_similarity(a: str, b: str) -> float:
    A, B = _token_set(a), _token_set(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    j = inter / union
    # Small bonus if house numbers match
    numsA, numsB = _num_set(a), _num_set(b)
    num_bonus = 0.15 if numsA and numsB and (numsA & numsB) else 0.0
    return min(1.0, j + num_bonus)

def address_match_score(ocr, id_back_b: Optional[bytes], bill_b: Optional[bytes]) -> Tuple[float, Dict[str, Any]]:
    """Use OCR adapter's raw_text path to compare addresses.
    NOTE: OCR.extract(front, back) appends raw text of provided images; it never parses address itself
    (only name/dob/id + raw_text), so we parse address here.
    """
    def _raw(front=None, back=None) -> str:
        if not (front or back):
            return ""
        try:
            _, fields = ocr.extract(front, back)
            return str(fields.get("raw_text", ""))
        except Exception:
            return ""

    raw_id_back = _raw(back=id_back_b)
    raw_bill    = _raw(front=bill_b)

    id_addr   = _extract_address_text(raw_id_back, "id")
    bill_addr = _extract_address_text(raw_bill, "bill")

    score = address_similarity(id_addr, bill_addr)
    details = {
        "id_address": id_addr,
        "bill_address": bill_addr,
        "id_raw_len": len(raw_id_back),
        "bill_raw_len": len(raw_bill),
    }
    return score, details
