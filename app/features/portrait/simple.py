from typing import Tuple, Dict, Optional
from app.core.config import get_settings
from app.utils.portrait import detect_portrait

class SimplePortraitExtractor:
    def extract(self, front: bytes) -> Tuple[Optional[bytes], Dict]:
        settings = get_settings()
        crop, info = detect_portrait(
            front,
            min_size=settings.portrait_min_size,
            margin=settings.portrait_margin
        )
        if crop is None:
            return None, {"method": info.get("method"), "reason": info.get("reason", "not_found")}
        return crop, {"method": info.get("method"), "bbox": info.get("bbox")}
