# app/features/liveness/heuristic.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class HeuristicConfig:
    # Tunables; you can surface these as APP_* envs later if needed
    edge_weight: float = 0.45
    texture_weight: float = 0.35
    sharp_weight: float = 0.20

    # Normalization caps to keep metrics stable across devices
    edge_cap: float = 0.25      # max edge density (25% of pixels)
    sharp_cap: float = 500.0    # Laplacian variance cap
    tex_cap: float = 0.20       # HF energy cap (fraction of spectrum)


class HeuristicLivenessDetector:
    """
    Very lightweight, passive liveness score in [0,1] using:
      • Edge density (Canny)       -> printed attacks often have low/high extremes
      • High-frequency energy (FFT)-> photos/screens can attenuate micro-texture
      • Sharpness (Laplacian var)  -> out-of-focus / screen re-captures are softer
    Not a substitute for a production anti-spoof model; good as a baseline.
    """

    def __init__(self, cfg: Optional[HeuristicConfig] = None) -> None:
        self.cfg = cfg or HeuristicConfig()
        # Threshold for is_live; default to APP_LIVE_THRESHOLD if present, else 0.80
        try:
            self.default_threshold = float(os.getenv("APP_LIVE_THRESHOLD", "0.80"))
        except Exception:
            self.default_threshold = 0.80

    # ---- public API (matches your SimpleLivenessDetector style) ----
    def score(self, bgr: np.ndarray) -> float:
        """Return liveness score in [0,1]."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1) Edge density
        med = float(np.median(gray))
        low = int(max(0, 0.66 * med))
        high = int(min(255, 1.33 * med))
        edges = cv2.Canny(gray, low, high)
        edge_density = float(np.mean(edges > 0))  # [0,1]
        edge_feat = min(edge_density, self.cfg.edge_cap) / self.cfg.edge_cap  # [0,1]

        # 2) High-frequency energy (FFT magnitude outside center)
        # downscale for speed/stability
        small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        f = np.fft.fft2(small)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)

        # build a circular low-pass mask to isolate HF outside the center
        h, w = small.shape
        cy, cx = h // 2, w // 2
        radius = int(min(h, w) * 0.10)  # keep center 10% radius as LF
        yy, xx = np.ogrid[:h, :w]
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) > radius * radius
        hf_energy = float(np.mean(mag[mask]))
        # normalize by mean magnitude to be exposure-invariant
        mean_mag = float(np.mean(mag) + 1e-6)
        hf_ratio = hf_energy / mean_mag
        tex_feat = min(hf_ratio, self.cfg.tex_cap) / self.cfg.tex_cap  # [0,1]

        # 3) Sharpness via Laplacian variance
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharp_feat = min(sharp, self.cfg.sharp_cap) / self.cfg.sharp_cap  # [0,1]

        # Weighted sum -> score
        score = (
            self.cfg.edge_weight * edge_feat
            + self.cfg.texture_weight * tex_feat
            + self.cfg.sharp_weight * sharp_feat
        )
        # Clamp
        return float(max(0.0, min(1.0, score)))

    def is_live(self, bgr: np.ndarray, threshold: Optional[float] = None) -> bool:
        thr = self.default_threshold if threshold is None else float(threshold)
        return self.score(bgr) >= thr
