# bank-service

FastAPI AI/KYC service for bank-core. The service exposes
the existing KYC and health routes under its FastAPI application; it is versioned
independently from the Spring Boot backend and web/mobile clients.

## Requirements

- Python 3.11 or a compatible Python 3 release
- Dependencies from `requirements.txt`

The default configuration selects portable simple/heuristic backends. Optional
model-backed implementations require their own model files and runtime packages;
model binaries are intentionally not part of this repository.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # optional
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The Spring Boot backend reaches this service through its configured HTTP service
URL (for example `ML_BASE_URL`); no sibling-directory assumption is required.

## Verification

```bash
pytest -q
```

The repository also includes `Makefile` targets for `run`, `test`, and `sanity`.
The optional model backends remain environment-dependent when their model files
are not installed.

## Related repositories

- [bank-core](https://github.com/gimesha-adikari/bank-core) — Spring Boot authoritative banking API
- [bank-web](https://github.com/gimesha-adikari/bank-web) — Next.js web client
- [bank-app](https://github.com/gimesha-adikari/bank-app) — Android client
