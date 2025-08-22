from typing import Optional, Tuple, Dict
from io import BytesIO
from PIL import Image, ImageFilter, ImageOps
import pytesseract, re

def _decode_rgb(b: Optional[bytes]) -> Optional[Image.Image]:
    if b is None:
        return None
    try:
        return Image.open(BytesIO(b)).convert("RGB")
    except Exception:
        return None

def _preprocess_for_ocr(img: Image.Image) -> Image.Image:

    gray = img.convert("L")
    big = gray.resize((gray.width * 3, gray.height * 3), Image.Resampling.LANCZOS)
    big = ImageOps.autocontrast(big)
    big = big.filter(ImageFilter.SHARPEN)
    return big

def ocr_text(front_b: Optional[bytes], back_b: Optional[bytes]) -> str:
    parts = []
    for b in (front_b, back_b):
        if b:
            img = _decode_rgb(b)
            if img is not None:
                proc = _preprocess_for_ocr(img)
                txt = pytesseract.image_to_string(proc, lang="eng", config="--oem 3 --psm 6")
                parts.append(txt)
    return "\n".join(parts).strip()

def _normalize_text(t: str) -> str:
    t = re.sub(r"[·•]", ".", t)
    # Normalize ID label forms to 'ID: '
    t = re.sub(r"\bID[\s\.:]+", "ID: ", t, flags=re.IGNORECASE)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    return t

def parse_id_fields(text: str) -> Dict[str, Optional[str]]:
    t = _normalize_text(text)
    doc_raw = re.search(r"\b([A-Z]{1,3}\s*\d{6,10})\b", t)
    doc_num = re.sub(r"\s+", "", doc_raw.group(1)) if doc_raw else None

    dob = re.search(r"\b(\d{4}[-/\.]\d{2}[-/\.]\d{2})\b", t)

    name_m = re.search(r"Name[:\s]+([A-Za-z ,.'-]{3,})", t)

    return {
        "docNumber": doc_num,
        "dob": dob.group(1) if dob else None,
        "name": name_m.group(1).strip() if name_m else None,
        "textPreview": text[:300]
    }

def quality_score(fields: Dict[str, Optional[str]], text: str) -> float:
    hits = sum(fields[k] is not None for k in ["docNumber", "dob", "name"])
    len_bonus = 0.05 if len(text) >= 30 else 0.0
    return float(min(1.0, 0.6 + 0.1 * hits + len_bonus))
