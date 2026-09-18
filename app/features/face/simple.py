from typing import Any, Tuple, Dict
from PIL import Image
import numpy as np
from app.utils.images import decode_image, perceptual_similarity

class SimpleFaceMatcher:
    @staticmethod
    def _to_image(value: Any) -> Image.Image | None:
        if isinstance(value, Image.Image):
            return value.convert("L")
        if isinstance(value, (bytes, bytearray, memoryview)):
            return decode_image(bytes(value))
        if isinstance(value, np.ndarray):
            try:
                array = value
                if array.ndim == 3 and array.shape[-1] == 3:
                    array = array[:, :, ::-1]
                return Image.fromarray(array).convert("L")
            except Exception:
                return None
        return None

    def score(self, selfie: Any, doc_front: Any) -> Tuple[float, Dict]:
        s_img = self._to_image(selfie)
        f_img = self._to_image(doc_front)
        if s_img is None or f_img is None:
            return 0.0, {"error": "decode_failed"}
        score = perceptual_similarity(s_img, f_img, hash_size=16, dhash_weight=0.30)
        return float(score), {"algo": "dHash+L1", "hash_size": 16, "dhash_weight": 0.30}
