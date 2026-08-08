.PHONY: install data train evaluate monitor serve test lint format clean pipeline docker-build docker-run sast sca security

install:
	uv sync --extra dev --extra serving --extra tracking --extra security

data:
	python scripts/00_generate_sample_data.py
	python scripts/01_process_data.py

train:
	python scripts/02_train_model.py

evaluate:
	python scripts/03_evaluate_model.py

monitor:
	python scripts/05_refresh_monitor.py

serve:
	uvicorn pm_mlops.serving.api:app --reload --port 8000

pipeline: data train evaluate

test:
	pytest

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

# SAST: static analysis for security anti-patterns in our own code.
sast:
	bandit -c pyproject.toml -r src/ scripts/

# SCA: known-vulnerability scan of the resolved dependency set.
sca:
	pip-audit --desc --strict

security: sast sca

docker-build:
	docker build -t pm-mlops:latest .

docker-run:
	docker run --rm -p 8000:8000 pm-mlops:latest

clean:
	rm -rf data/processed models/*.joblib monitoring_reports .pytest_cache .ruff_cache **/__pycache__ *.egg-info htmlcov .coverage bandit-report.json
