# Search Engine Notebooks

This directory contains interactive demonstrations of the search engine prototype.

## Files

- **`search_engine_demo.ipynb`** - Interactive Jupyter notebook demonstrating all search engine features
- **`test_search_engine.py`** - Simple Python script to validate functionality

## Quick Start

### Option 1: Run the Test Script
```bash
cd notebooks
uv run python test_search_engine.py
```

### Option 2: Use the Jupyter Notebook
```bash
# Start Jupyter from the project root
uv run jupyter lab

# Then open notebooks/search_engine_demo.ipynb
```

## Features Demonstrated

✅ **BM25 Keyword Relevance** - Core text matching with field boosts  
✅ **Synonym Expansion** - "cv" → "resume" with lower weight  
✅ **Facet-Based Boosting** - Size, style, color queries get score boosts  
✅ **Popularity Signals** - Well-liked items get small ranking nudges  
✅ **MMR Diversity** - Final results are diverse, not near-duplicates  
✅ **Sparse Optimization** - Efficient TF-IDF with precomputed norms  

## Example Queries to Try

- `"resume"` - Basic keyword search
- `"cv"` - Synonym expansion test  
- `"A4 minimal template"` - Facet boosting demo
- `"gold logo modern"` - Multi-term with facets
- `"instagram minimal white"` - Social media template search
- `"presentation slides business"` - Business content search

## Interactive Features

The notebook includes:
- Step-by-step pipeline analysis
- Configuration experiments (MMR lambda tuning)
- Performance insights (sparse TF-IDF optimization)
- Diversity comparison (with/without MMR)
- Real-time search interface

## Requirements

- The search engine is implemented in `../src/search_engine.py`
- Uses only Python standard library + `rank_bm25` + `scikit-learn`
- No external data files needed - everything is self-contained
