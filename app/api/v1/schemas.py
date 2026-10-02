from __future__ import annotations

from typing import Optional, Dict, Any, List
from uuid import UUID
from pydantic import BaseModel, Field, Base64Bytes

class KycPayload(BaseModel):
    bankUserId: Optional[UUID] = Field(default=None, description="trusted bank-core subject")
    selfie: Optional[Base64Bytes] = Field(default=None, description="base64-encoded selfie image")
    docFront: Optional[Base64Bytes] = Field(default=None, description="base64-encoded ID front (legacy)")
    docFrontImage: Optional[Base64Bytes] = Field(default=None, description="base64-encoded ID front")
    docBack: Optional[Base64Bytes] = Field(default=None, description="base64-encoded ID back (legacy)")
    docBackImage: Optional[Base64Bytes] = Field(default=None, description="base64-encoded ID back")
    docPortraitImage: Optional[Base64Bytes] = Field(default=None, description="base64-encoded portrait crop from document")
    addressProofImage: Optional[Base64Bytes] = Field(default=None, description="base64-encoded proof of address")
    meta: Optional[Dict[str, str]] = Field(default=None, description="optional country/document hints")

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
