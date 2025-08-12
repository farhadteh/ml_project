# Search Engine Project

A complete single-file search engine prototype demonstrating modern ranking techniques including BM25, synonym expansion, facet boosting, and MMR diversity re-ranking.

## 🎯 Project Overview

This project implements a fast, relevant, and intelligent search engine for creative assets (like Canva templates). It features:

- **BM25 keyword relevance** with field boosts (title, tags, description)
- **Synonym expansion** (e.g., "cv" → "resume") with configurable weights
- **Facet-based boosting** for size, style, and color queries
- **Popularity signals** with normalized scoring
- **MMR diversity re-ranking** using optimized sparse TF-IDF
- **Pure functional design** with comprehensive test coverage

## 🚀 Quick Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### 1. Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: pip install uv
```

### 2. Clone and Setup

```bash
git clone <your-repo-url>
cd ml_project

# Install Python 3.12.8 via uv
uv python install 3.12.8

# Create virtual environment
uv venv

# Install core dependencies
uv sync

# Install development tools
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install
```

### 3. Test the Search Engine

```bash
# Quick validation
cd examples
uv run python test_search_engine.py

# Run comprehensive tests
uv run pytest

# Interactive Jupyter demo
uv run jupyter lab
# Then open: notebooks/search_engine_demo.ipynb
```

## 📦 Dependencies

### Core Dependencies (Always Installed)
- **Search & ML**: `rank-bm25`, `scikit-learn`, `spacy`
- **Data Processing**: `numpy`, `pandas`
- **Testing**: `pytest`
- **Notebooks**: `jupyter`

### Optional Dependencies

#### Deep Learning Group (Optional)
```bash
# Only install if you need PyTorch/Transformers
uv sync --group dl
```
- `torch`, `torchvision`, `transformers`, `sentence-transformers`

#### Development Tools (Recommended)
```bash
uv sync --group dev
```
- `black`, `ruff`, `isort`, `mypy`, `pre-commit`

## 🔍 Usage Examples

### Basic Search
```python
from src.search_engine import search, create_search_context, CONFIG, DOCS

# Initialize once
ctx = create_search_context(DOCS, CONFIG)

# Search queries
results = search("resume template", ctx, CONFIG, k=5)
for doc_id, score in results:
    print(f"{doc_id}: {score:.3f}")
```

### Advanced Features
```python
# Synonym expansion: "cv" finds "resume" documents
results = search("cv", ctx, CONFIG, k=3)

# Facet boosting: boost documents with specific attributes
results = search("A4 minimal template", ctx, CONFIG, k=5)

# Empty query: returns by popularity
results = search("", ctx, CONFIG, k=3)
```

## �� Project Structure

```
ml_project/
├── src/
│   └── search_engine.py      # Complete search engine (727 lines)
├── tests/
│   └── test_search_engine.py # Comprehensive test suite (182 lines)
├── notebooks/
│   ├── search_engine_demo.ipynb # Interactive demo
│   └── README.md
├── examples/
│   └── test_search_engine.py # Quick validation script
├── docs/
│   ├── project_plan.md       # Original project specification
│   └── implementation_plan.md # Detailed implementation steps
└── pyproject.toml            # Dependencies and configuration
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_search_engine.py

# Check code formatting
uv run pre-commit run --all-files
```

## 💻 Development

### Code Quality
```bash
# Format code
uv run black .
uv run isort .

# Lint code
uv run ruff check . --fix

# Type checking
uv run mypy src/

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

### Making Changes
1. Make your changes
2. Run tests: `uv run pytest`
3. Format code: `uv run pre-commit run --all-files`
4. Commit with conventional commit messages

## 🔧 Configuration

The search engine is configured via `CONFIG` dictionary in `src/search_engine.py`:

```python
CONFIG = {
    "k1": 1.2,                    # BM25 parameter
    "b": 0.75,                    # BM25 parameter
    "field_boosts": {             # Field importance weights
        "title": 2.0,
        "tags": 1.5,
        "desc": 1.0
    },
    "synonyms": {                 # Query expansion
        "cv": ["resume"],
        "photo": ["image", "picture"]
    },
    "syn_weight": 0.7,           # Synonym weight (< 1.0)
    "facet_boost": 2.0,          # Facet match bonus
    "pop_weight": 0.1,           # Popularity influence
    "mmr_lambda": 0.7            # Diversity vs relevance (0-1)
}
```

## 🚨 Troubleshooting

### Common Issues

**Import Errors**: Make sure you're in the project root and have run `uv sync`

**Missing spaCy Model**: The search engine uses `spacy.blank("en")` which doesn't require model downloads

**Platform Compatibility**: Heavy ML dependencies (PyTorch) are optional. Core search works on all platforms.

**Pre-commit Failures**: Run `uv run pre-commit run --all-files` to fix formatting issues

### Platform-Specific Notes

**macOS x86_64**: Some PyTorch versions may not be available. The core search engine works without them.

**Windows**: Make sure to use PowerShell or Command Prompt with proper path settings.

**Linux**: All dependencies should work out of the box.

## 📚 Documentation

- **Interactive Demo**: `notebooks/search_engine_demo.ipynb`
- **Project Plan**: `docs/project_plan.md`
- **Implementation Details**: `docs/implementation_plan.md`
- **API Reference**: Docstrings in `src/search_engine.py`

## 📈 Performance

- **BM25 Scoring**: O(N·|q|) where N=documents, |q|=query terms
- **MMR Selection**: O(k·N) with sparse cosine similarity
- **Memory**: Efficient sparse TF-IDF with precomputed document norms
- **Typical Performance**: Sub-millisecond for 10-100 documents

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test: `uv run pytest`
4. Format code: `uv run pre-commit run --all-files`
5. Commit: `git commit -m 'feat: Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
