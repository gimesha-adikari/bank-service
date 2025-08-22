from fastapi import FastAPI
from .config import Settings

def mount_features(app: FastAPI, settings: Settings) -> None:
    # KYC
    if settings.enable_feature_kyc:
        from app.api.v1.kyc import router as kyc_router
        app.include_router(kyc_router, prefix=settings.api_prefix)
    # Chatbot
    if settings.enable_feature_chatbot:
        from app.api.v1.chatbot import router as chatbot_router
        app.include_router(chatbot_router, prefix=settings.api_prefix)
    # Loans
    if settings.enable_feature_loans:
        from app.api.v1.loans import router as loans_router
        app.include_router(loans_router, prefix=settings.api_prefix)
