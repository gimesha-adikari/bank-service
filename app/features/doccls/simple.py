from typing import Tuple, Dict, Optional
from app.utils.doc_class import classify_doc

class SimpleDocClassifier:
    def classify(self, front: Optional[bytes], back: Optional[bytes]) -> Tuple[float, Dict]:
        score, cls, details = classify_doc(front, back)
        if cls and "class" not in details:
            details["class"] = cls
        return float(score), details
