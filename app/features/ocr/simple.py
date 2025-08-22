from typing import Tuple, Dict, Optional
from app.utils.ocr_simple import ocr_text, parse_id_fields, quality_score

class SimpleOcrEngine:
    def extract(self, front: Optional[bytes], back: Optional[bytes]) -> Tuple[float, Dict]:
        text = ocr_text(front, back)
        if not text:
            return 0.0, {"error": "no_text"}
        fields = parse_id_fields(text)
        score  = quality_score(fields, text)
        return float(score), fields
