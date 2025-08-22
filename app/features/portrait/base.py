from typing import Protocol, Tuple, Dict, Optional

class PortraitExtractor(Protocol):
    def extract(self, front: bytes) -> Tuple[Optional[bytes], Dict]:
        """
        Returns (portrait_bytes or None, details dict).
        details may include: bbox, method, reason.
        """
        ...
