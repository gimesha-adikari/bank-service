from typing import Tuple, Dict
from app.utils.images import decode_image, perceptual_similarity

class SimpleFaceMatcher:
    def score(self, selfie: bytes, doc_front: bytes) -> Tuple[float, Dict]:
        s_img = decode_image(selfie)
        f_img = decode_image(doc_front)
        if s_img is None or f_img is None:
            return 0.0, {"error": "decode_failed"}
        score = perceptual_similarity(s_img, f_img, hash_size=16, dhash_weight=0.30)
        return float(score), {"algo": "dHash+L1", "hash_size": 16, "dhash_weight": 0.30}
