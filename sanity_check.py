import os, sys, importlib

PACKAGES = [
    ("fastapi", "__version__"),
    ("uvicorn", "__version__"),
    ("pydantic", "__version__"),
    ("numpy", "__version__"),
    ("cv2", "__version__"),
    ("onnxruntime", "__version__"),
    ("PIL", "__version__"),
    ("pytesseract", "__version__"),
    ("httpx", "__version__"),
]

print("Python:", sys.version.replace("\n", " "))
for mod_name, attr in PACKAGES:
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, attr, None)
        label = "Pillow" if mod_name == "PIL" else ("opencv-python" if mod_name == "cv2" else mod_name)
        print(f"{label}: {ver}")
    except Exception as e:
        print(f"{mod_name}: (import failed) {e}")

try:
    import pytesseract
    print("tesseract version:", pytesseract.get_tesseract_version())
except Exception as e:
    print("tesseract check failed:", e)

try:
    import httpx
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))
    url = f"http://{host}:{port}/api/v1/kyc/ping"
    with httpx.Client(timeout=2.0) as client:
        r = client.get(url)
        print("/api/v1/kyc/ping:", r.status_code, r.text[:120])
        assert r.status_code == 200
        print("API OK ✅")
except Exception as e:
    print("API ping failed:", e)
    print("(Start the server with: uvicorn app.main:app --host 0.0.0.0 --port 8000)")
