from typing import Tuple, Dict
from app.utils.liveness import liveness_score
from app.utils.images import decode_image

class SimpleLivenessDetector:
    def score(self, selfie: bytes) -> Tuple[float, Dict]:
        s_img = decode_image(selfie)
        if s_img is None:
            return 0.0, {"error": "decode_failed"}
        score, extra = liveness_score(s_img)
        return float(score), extra
