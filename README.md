# ML Project Template

A pragmatic ML project scaffold with uv, pytest, and pre-commit. Uses a src/ layout and Makefile.

## Quickstart

1) Install Python 3.12.8

```bash
uv python install 3.12.8
```

2) Create and activate a virtual environment

```bash
uv venv
# optional activation
source .venv/bin/activate
```

3) Install dependencies

```bash
uv sync          # runtime deps
uv sync --group dev  # dev tools (formatters, linters, mypy)
```

4) Install package in editable mode

```bash
uv pip install -e .
```

5) Install pre-commit hooks

```bash
uv run pre-commit install
```

## Common tasks

- Run tests:
  ```bash
  uv run pytest
  ```
- Run all hooks on all files:
  ```bash
  uv run pre-commit run --all-files
  ```
- Format and lint:
  ```bash
  make format && make lint
  ```
- Example data and training entrypoints:
  ```bash
  make data
  make train
  ```

## Project structure

```
./
├── LICENSE
├── README.md
├── Makefile
├── configs/
│   └── model1.yaml
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── docs/
├── models/
├── notebooks/
├── references/
├── reports/
│   └── figures/
├── requirements.txt
└── src/
    ├── __init__.py
    ├── data/
    │   ├── build_features.py
    │   ├── cleaning.py
    │   ├── ingestion.py
    │   ├── labeling.py
    │   ├── splitting.py
    │   └── validation.py
    ├── models/
    │   └── model1/
    │       ├── dataloader.py
    │       ├── hyperparameters_tuning.py
    │       ├── model.py
    │       ├── predict.py
    │       ├── preprocessing.py
    │       └── train.py
    └── visualization/
        ├── evaluation.py
        └── exploration.py
```

## Pre-commit Checks

Run these checks manually before committing to avoid having your commits rejected:

1. **Run all pre-commit hooks on all files**:
   ```bash
   uv run pre-commit run --all-files
   ```

2. **Run pre-commit on only staged files** (what would happen during commit):
   ```bash
   uv run pre-commit run
   ```

3. **Run specific hooks**:
   ```bash
   uv run pre-commit run black --all-files
   uv run pre-commit run ruff --all-files
   uv run pre-commit run isort --all-files
   ```

4. **Run individual tools directly**:
   ```bash
   uv run black .
   uv run ruff check . --fix
   uv run isort .
   ```

Common fixes for pre-commit issues:
- Fix import ordering: `uv run isort .`
- Fix ruff linting issues: `uv run ruff check . --fix`

## Notes
- Dev tools live in a dependency group; install them with `uv sync --group dev`.
- Pre-commit manages its own hook environments; having tools in the venv speeds up runs but is optional.
- Python path includes `src` for tests (see `pyproject.toml`).

additinal libries:

Install PyTorch CPU-only (Recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers

Install FAISS (if needed)
# Install SWIG first
brew install swig
# Then install faiss
pip install faiss-cpu
