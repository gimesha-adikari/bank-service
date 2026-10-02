from __future__ import annotations

from functools import lru_cache
import os
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Bank AI Service"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    bank_service_auth_secret: str = Field(validation_alias="BANK_SERVICE_AUTH_SECRET")

    # feature flags
    enable_feature_kyc: bool = True
    enable_feature_chatbot: bool = False
    enable_feature_loans: bool = False

    # thresholds for KYC
    face_threshold: float = 0.85
    live_threshold: float = 0.80
    ocr_threshold: float  = 0.80
    doc_threshold: float = 0.80
    address_threshold: float = 0.75

    # backend selectors
    face_backend: str = "simple"
    liveness_backend: str = "simple"
    # Compatibility setting retained for older local .env files.
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

    @model_validator(mode="after")
    def validate_service_auth_secret(self) -> "Settings":
        secret = self.bank_service_auth_secret
        if not secret or not secret.strip():
            raise ValueError("BANK_SERVICE_AUTH_SECRET must be configured")
        if secret.strip() in {
            "CHANGE_ME",
            "CHANGE_ME_TO_A_RANDOM_SECRET",
            "local-development-secret-change-me-before-sharing",
        }:
            raise ValueError("BANK_SERVICE_AUTH_SECRET uses a forbidden placeholder")
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("BANK_SERVICE_AUTH_SECRET must be at least 32 UTF-8 bytes")
        return self

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
