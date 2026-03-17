venv:
	python3 -m venv .venv

install: venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e ".[dev]"

smoke:
	.venv/bin/python scripts/smoke_check.py

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff format .
