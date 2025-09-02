from __future__ import annotations
from fastapi import FastAPI
from app.api.debug_router import router as debug_router

app = FastAPI(title="Bank AI – Debug API", version="0.1.0")
app.include_router(debug_router)

@app.get("/")
def root():
    return {"hello": "debug", "routes": ["/debug/ping", "/debug/face-score", "/debug/liveness", "/debug/ocr", "/debug/doc-classify"]}
