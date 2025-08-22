from typing import Protocol, Tuple, Dict

class LivenessDetector(Protocol):
    def score(self, selfie: bytes) -> Tuple[float, Dict]: ...
