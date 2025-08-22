from typing import Protocol, Tuple, Dict

class FaceMatcher(Protocol):
    def score(self, selfie: bytes, doc_front: bytes) -> Tuple[float, Dict]: ...
