from typing import Protocol, Tuple, Dict, Optional

class DocClassifier(Protocol):
    def classify(self, front: Optional[bytes], back: Optional[bytes]) -> Tuple[float, Dict]: ...
