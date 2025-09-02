.PHONY: run test sanity freeze

run:
	uvicorn app.main:app --host $${APP_HOST:-0.0.0.0} --port $${APP_PORT:-8000} --reload

test:
	pytest -q

sanity:
	python sanity_check.py

freeze:
	python -m pip freeze > requirements.txt
