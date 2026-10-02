from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

from app.core.config import get_settings


def require_bank_core_auth(
    credential: str | None = Header(default=None, alias="X-Bank-Core-Auth"),
) -> None:
    expected = get_settings().bank_service_auth_secret
    if credential is None or not secrets.compare_digest(credential, expected):
        raise HTTPException(status_code=401, detail="Invalid internal service credentials")
