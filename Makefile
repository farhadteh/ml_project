.PHONY: help install venv sync dev test lint format precommit-install precommit data train clean

help:
	@echo "Targets: install | venv | sync | dev | test | lint | format | precommit-install | precommit | data | train | clean"

install: venv sync ## Create venv and install deps

venv:
	uv venv

sync:
	uv sync

# Install runtime + dev tools
dev:
	uv sync --group dev

# Run tests
test:
	uv run pytest

# Lint and type-check
lint:
	uv run ruff . && uv run mypy

# Auto-format
format:
	uv run black . && uv run isort .

# Install and run pre-commit
precommit-install:
	uv run pre-commit install

precommit:
	uv run pre-commit run --all-files

# Example workflow shortcuts
data:
	uv run python src/data/ingestion.py

train:
	uv run python src/models/model1/train.py

clean:
	rm -rf .pytest_cache .mypy_cache ruff_cache **/__pycache__ build dist *.egg-info
