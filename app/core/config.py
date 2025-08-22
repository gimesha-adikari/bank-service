from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Bank AI Service"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"

    # feature flags
    enable_feature_kyc: bool = True
    enable_feature_chatbot: bool = False
    enable_feature_loans: bool = False

    # thresholds for KYC
    face_threshold: float = 0.85
    live_threshold: float = 0.80
    ocr_threshold: float  = 0.80
    doc_threshold: float = 0.80

    # backend selectors
    face_backend: str = "simple"
    live_backend: str = "simple"
    ocr_backend: str = "simple"
    doc_backend: str = "simple"

    # portrait extraction
    doc_portrait_mode: str = "auto"
    portrait_backend: str = "simple"
    portrait_min_size: int = 64
    portrait_margin: float = 0.20

    # calibration logging
    calibration_log: bool = False
    calibration_dir: str = "./calib"
    instance_id: str | None = None

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        extra="ignore",
    )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

def segmented_threshold(prefix: str, country: Optional[str], doc: Optional[str], default: float) -> float:
    keys = []
    if country and doc:
        keys.append(f"{prefix}__{country}__{doc}")
    if country:
        keys.append(f"{prefix}__{country}")
    for k in keys:
        v = os.getenv(k.upper())
        if v:
            try:
                return float(v)
            except Exception:
                pass
    return default