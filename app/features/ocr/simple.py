from typing import Tuple, Dict, Optional
from app.utils.ocr_simple import ocr_text, parse_id_fields, quality_score


class SimpleOcrEngine:
    """
    Minimal OCR engine backed by pytesseract with light preprocessing.
    IMPORTANT: Always returns 'raw_text' so downstream code can regex scan.
    """

    def extract(self, front: Optional[bytes], back: Optional[bytes]) -> Tuple[float, Dict]:
        text = ocr_text(front, back)
        if not text:
            return 0.0, {"error": "no_text", "raw_text": ""}

        fields = parse_id_fields(text)

        # Guarantee 'raw_text' for downstream fallbacks, even if parser changes later.
        if "raw_text" not in fields:
            fields["raw_text"] = text

        score = quality_score(fields, text)
        return float(score), fields
