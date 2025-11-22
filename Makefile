.PHONY: install lint format test seed index eval api ui docker clean

install:
	python -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -e ".[dev]"
	.venv/bin/pre-commit install

lint:
	.venv/bin/ruff check src tests
	.venv/bin/mypy src

format:
	.venv/bin/ruff format src tests
	.venv/bin/ruff check --fix src tests

test:
	.venv/bin/pytest -q

seed:
	.venv/bin/python scripts/seed_data.py

index:
	.venv/bin/python -m matcher.pipeline.index

eval:
	OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m matcher.eval.eval_runner

api:
	.venv/bin/uvicorn matcher.api.main:app --reload --port 8000

ui:
	.venv/bin/streamlit run src/matcher/ui/app.py

docker:
	docker build -t resume-jd-matcher:latest .

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache __pycache__ build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
