.PHONY: setup test lint typecheck fmt reproduce clean help

UV ?= uv
PY := $(UV) run python

help:
	@echo "setup      install pinned deps into .venv (uv)"
	@echo "test       run pytest with coverage gate on core logic"
	@echo "lint       ruff check + format check"
	@echo "typecheck  mypy --strict"
	@echo "fmt        ruff format + autofix"
	@echo "reproduce  regenerate every number and figure in the README from scratch"
	@echo "clean      remove generated results and caches"

setup:
	$(UV) venv --python 3.12
	$(UV) pip install -e ".[dev]"

test:
	$(UV) run pytest --cov=lobsim --cov-report=term-missing --cov-fail-under=85

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

typecheck:
	$(UV) run mypy

fmt:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

# Full pipeline. Each stage writes raw JSON to results/raw/; the last stage renders
# the tables that README.md embeds. Ordering matters: validation -> backtests -> stats -> figures.
reproduce:
	$(PY) scripts/run_validation.py
	$(PY) scripts/train_policy.py
	$(PY) scripts/run_backtests.py
	$(PY) scripts/run_ablations.py
	$(PY) scripts/analyse_results.py
	$(PY) scripts/make_figures.py
	$(PY) scripts/render_readme_tables.py
	@echo "reproduce: complete. See results/raw and results/figures."

clean:
	rm -rf results/raw/* results/figures/* .pytest_cache .mypy_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
