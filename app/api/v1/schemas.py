from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, Base64Bytes

class KycPayload(BaseModel):
    selfie: Optional[Base64Bytes] = Field(default=None, description="base64-encoded selfie image")
    docFront: Optional[Base64Bytes] = Field(default=None, description="base64-encoded ID front (legacy)")
    docFrontImage: Optional[Base64Bytes] = Field(default=None, description="base64-encoded ID front")
    docBackImage: Optional[Base64Bytes] = Field(default=None, description="base64-encoded ID back")
    docPortraitImage: Optional[Base64Bytes] = Field(default=None, description="base64-encoded portrait crop from document")

class KycResult(BaseModel):
    type: str
    score: float | None = None
    passed: bool | None = None
    detailsJson: str | None = None

class CheckResult(BaseModel):
    type: str
    score: float | None
    passed: bool | None
    details: Dict[str, Any] | None = None

class AggregateResponse(BaseModel):
    decision: str
    reasons: List[str] = []
    checks: list[CheckResult]