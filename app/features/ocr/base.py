from typing import Protocol, Tuple, Dict, Optional

class OcrEngine(Protocol):
    def extract(self, front: Optional[bytes], back: Optional[bytes]) -> Tuple[float, Dict]: ...
