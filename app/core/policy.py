from typing import List, Tuple, Dict, Any
from app.core.config import Settings

Decision = str  # "APPROVE" | "UNDER_REVIEW" | "REJECT"

def decide(settings: Settings, checks: Dict[str, Dict[str, Any]]) -> Tuple[Decision, List[str]]:
    """
    checks: {
      "FACE_MATCH": {"score": float, "passed": bool, "details": {...}},
      "LIVENESS": {"score": float, "passed": bool, "details": {...}},
      "OCR_ID": {"score": float, "passed": bool, "details": {...}},
      "DOC_CLASS": {"score": float, "passed": bool, "details": {...}}
    }
    """
    reasons: List[str] = []

    if not checks.get("DOC_CLASS", {}).get("passed"):
        reasons.append("document_type_unconfirmed")

    if not checks.get("OCR_ID", {}).get("passed"):
        reasons.append("ocr_quality_insufficient")

    if not checks.get("FACE_MATCH", {}).get("passed"):
        reasons.append("face_match_below_threshold")

    if not checks.get("LIVENESS", {}).get("passed"):
        reasons.append("liveness_check_failed")

    if not reasons:
        return "APPROVE", reasons

    fm = checks.get("FACE_MATCH", {}).get("score")
    lv = checks.get("LIVENESS", {}).get("score")
    if (fm is not None and fm < 0.40) or (lv is not None and lv < 0.40):
        return "REJECT", reasons

    return "UNDER_REVIEW", reasons
