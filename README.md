# Bank AI Service (FastAPI)

Modular service for ML/AI features:
- KYC (now)
- Chatbot (later)
- Loan approval (later)

## Run
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # optional
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
