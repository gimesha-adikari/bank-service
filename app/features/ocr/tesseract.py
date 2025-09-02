# app/features/ocr/tesseract.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

# Local field parsers
try:
    from app.features.ocr.fields.nic import parse_nic_text
except Exception:
    parse_nic_text = None  # type: ignore


@dataclass
class OcrConfig:
    langs: str = os.getenv("APP_OCR_LANGS", "eng")
    psm: int = int(os.getenv("APP_OCR_PSM", "6"))   # 6: Assume a single uniform block of text.
    oem: Optional[int] = int(os.getenv("APP_OCR_OEM", "3")) if os.getenv("APP_OCR_OEM") else None
    tess_cmd: Optional[str] = os.getenv("TESSERACT_CMD", None)
    upscale: float = float(os.getenv("APP_OCR_UPSCALE", "3.0"))  # upscale factor before binarization

    def to_args(self) -> str:
        args = f"--psm {self.psm}"
        if self.oem is not None:
            args += f" --oem {self.oem}"
        return args


class TesseractOcrEngine:
    """
    OCR engine using pytesseract. Provides:
      - text(bgr) -> raw text
      - extract_fields(bgr, country='LK', doc='NIC') -> parsed fields (best-effort)
    """

    def __init__(self, config: Optional[OcrConfig] = None) -> None:
        self.cfg = config or OcrConfig()
        if self.cfg.tess_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.cfg.tess_cmd

        # Quick availability check; don't hard-fail so service can still boot
        try:
            _ = pytesseract.get_tesseract_version()
        except Exception:
            pass

    # -------- Public API --------

    def text(self, bgr: np.ndarray) -> str:
        """
        Run OCR on a BGR image. Returns raw text (UTF-8).
        """
        pil = self._preprocess_for_ocr(bgr)
        txt = pytesseract.image_to_string(pil, lang=self.cfg.langs, config=self.cfg.to_args())
        return txt.strip()

    def extract_fields(self, bgr: np.ndarray, country: str = "LK", doc: str = "NIC") -> Dict[str, str]:
        """
        Best-effort structured extraction for supported (country, doc) pairs.
        Falls back to returning only the raw text if no parser is available.
        """
        raw = self.text(bgr)
        result: Dict[str, str] = {"raw_text": raw}

        key = f"{country.upper()}::{doc.upper()}"
        if key == "LK::NIC" and callable(parse_nic_text):
            try:
                parsed = parse_nic_text(raw)  # <-- FIX: call the function
                if isinstance(parsed, dict):
                    result.update(parsed)
            except Exception:
                # Non-fatal: keep raw_text only
                pass
        return result

    # -------- Internals --------

    def _preprocess_for_ocr(self, bgr: np.ndarray) -> Image.Image:
        """
        Stronger preprocessing for ID cards / printed text:
          - grayscale
          - upscale by APP_OCR_UPSCALE (default 3x) with cubic interp
          - light blur + unsharp
          - Otsu binarization
        Return a PIL Image compatible with pytesseract.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Upscale first to help OCR
        scale = max(1.0, float(self.cfg.upscale))
        up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Mild blur to reduce aliasing
        blur = cv2.GaussianBlur(up, (3, 3), 0)

        # Unsharp mask to crisp edges
        sharp = cv2.addWeighted(up, 1.5, blur, -0.5, 0)

        # Otsu threshold
        _, th = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Small median filter to clean salt-and-pepper artifacts
        th = cv2.medianBlur(th, 3)

        return Image.fromarray(th)
