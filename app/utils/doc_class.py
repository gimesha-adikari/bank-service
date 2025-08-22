from __future__ import annotations

from typing import Tuple, Dict
import re
from app.utils.ocr_simple import ocr_text

KEYS_NIC = [
    r"national identity", r"identity card", r"\bnic\b",
    r"department of registration", r"sri lanka"
]

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s

def classify_doc(front_b: bytes | None, back_b: bytes | None) -> Tuple[float, str, Dict]:
    text = ocr_text(front_b, back_b)
    if not text:
        return 0.0, "UNKNOWN", {"class": None, "country": None, "textPreview": ""}

    norm = _normalize(text)
    hits = sum(1 for pat in KEYS_NIC if re.search(pat, norm))

    score = min(1.0, 0.6 + 0.1 * hits)

    doc_class = "NIC" if hits >= 2 else "UNKNOWN"
    country = "LK" if "sri lanka" in norm else None

    details = {
        "class": doc_class if doc_class != "UNKNOWN" else None,
        "country": country,
        "hits": hits,
        "textPreview": text[:300]
    }
    return float(score), doc_class, details
