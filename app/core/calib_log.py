from __future__ import annotations
import os, csv, datetime as dt
from typing import Any, Dict, Optional

_HEADERS = [
    "ts", "request_id", "instance_id",
    "country", "doc_class",
    "face_score", "live_score", "ocr_score", "doc_score",
    "face_pass", "live_pass", "ocr_pass", "doc_pass",
    "decision", "reasons",
]

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _today_csv_path(dirpath: str) -> str:
    day = dt.datetime.utcnow().date().isoformat()
    return os.path.join(dirpath, f"{day}.csv")

def _maybe_write_header(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(_HEADERS)

def _bool_str(b: Optional[bool]) -> str:
    if b is None:
        return ""
    return "1" if b else "0"

def _num_str(x: Optional[float]) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.6f}"
    except Exception:
        return ""

def write_row(
    dirpath: str,
    request_id: str,
    instance_id: Optional[str],
    decision: str,
    reasons: list[str],
    checks: Dict[str, Dict[str, Any]],
    country: Optional[str],
    doc_class: Optional[str],
) -> None:
    """
    checks = {
      "FACE_MATCH": {"score": float|None, "passed": bool|None},
      "LIVENESS":   {"score": float|None, "passed": bool|None},
      "OCR_ID":     {"score": float|None, "passed": bool|None},
      "DOC_CLASS":  {"score": float|None, "passed": bool|None},
    }
    """
    _ensure_dir(dirpath)
    csv_path = _today_csv_path(dirpath)
    _maybe_write_header(csv_path)

    row = [
        dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        request_id,
        instance_id or "",
        (country or "UNK"),
        (doc_class or "UNK"),
        _num_str(checks.get("FACE_MATCH", {}).get("score")),
        _num_str(checks.get("LIVENESS",   {}).get("score")),
        _num_str(checks.get("OCR_ID",     {}).get("score")),
        _num_str(checks.get("DOC_CLASS",  {}).get("score")),
        _bool_str(checks.get("FACE_MATCH", {}).get("passed")),
        _bool_str(checks.get("LIVENESS",   {}).get("passed")),
        _bool_str(checks.get("OCR_ID",     {}).get("passed")),
        _bool_str(checks.get("DOC_CLASS",  {}).get("passed")),
        decision,
        ";".join(reasons),
    ]


    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
