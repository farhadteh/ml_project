"""
Fast-loading version of ranker with optimized imports.

This module demonstrates how to structure code to minimize import overhead
while maintaining full functionality when needed.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

# Configure environment for fast imports before any heavy imports
os.environ.update(
    {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    }
)

# Core functions that don't require heavy imports
from .ranker import (
    blend_scores,
    cosine_similarity,
    popularity_prior,
    recency_prior,
    tokenize,
    top_k,
)

# Type definitions
Document = dict[str, Any]
Weights = tuple[float, float, float]

# Global variables for lazy-loaded modules
_torch = None
_transformers = None
_sentence_transformers = None


def get_torch():
    """Lazy load PyTorch only when needed."""
    global _torch
    if _torch is None:
        print("Loading PyTorch...")
        import torch

        _torch = torch
    return _torch


def get_transformers():
    """Lazy load transformers only when needed."""
    global _transformers
    if _transformers is None:
        print("Loading transformers...")
        import transformers

        _transformers = transformers
    return _transformers


def get_sentence_transformers():
    """Lazy load sentence-transformers only when needed."""
    global _sentence_transformers
    if _sentence_transformers is None:
        print("Loading sentence-transformers...")
        import sentence_transformers

        _sentence_transformers = sentence_transformers
    return _sentence_transformers


def rank_fast(
    query: str,
    documents: Sequence[Document],
    k: int,
    weights: Weights = (1.0, 0.5, 0.2),
    query_embedding: Sequence[float] | None = None,
    use_semantic: bool = False,
) -> list[tuple[str, float]]:
    """
    Fast ranking function that only loads heavy libraries when semantic search is requested.

    Parameters
    ----------
    query : str
        Search query
    documents : Sequence[Document]
        Documents to rank
    k : int
        Number of results to return
    weights : Weights
        Weighting for (BM25, semantic, priors)
    query_embedding : Sequence[float] | None
        Pre-computed query embedding
    use_semantic : bool
        Whether to use semantic similarity (triggers heavy import)

    Returns
    -------
    list[tuple[str, float]]
        Top-k ranked results
    """
    if k <= 0 or not documents:
        return []

    query_tokens = tokenize(query)

    # Prepare document tokens
    corpus_tokens: list[list[str]] = []
    for doc in documents:
        if "tokens" in doc and isinstance(doc["tokens"], list):
            tokens: list[str] = [str(t) for t in doc["tokens"]]
        else:
            text = str(doc.get("text", ""))
            tokens = tokenize(text)
        corpus_tokens.append(tokens)

    # BM25 scores - only import rank_bm25 when needed
    if not corpus_tokens or not any(corpus_tokens):
        bm25_values = [0.0] * len(documents)
    else:
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi(corpus_tokens)
        bm25_values = (
            bm25.get_scores(query_tokens).tolist() if query_tokens else [0.0] * len(documents)
        )

    # Fallback for small corpora
    if query_tokens and all(score == 0.0 for score in bm25_values):
        query_set = set(query_tokens)
        bm25_values = [float(len(query_set.intersection(set(tokens)))) for tokens in corpus_tokens]

    # Semantic similarity - only compute if requested (and only load transformers then)
    semantic_values: list[float] = [0.0 for _ in documents]
    if use_semantic and (query_embedding is not None or any(doc.get("emb") for doc in documents)):
        # This will trigger the heavy import only when semantic search is actually used
        if query_embedding is not None:
            semantic_values = [
                (
                    cosine_similarity(query_embedding, doc.get("emb"))
                    if isinstance(doc.get("emb"), list)
                    else 0.0
                )
                for doc in documents
            ]

    # Priors computation (lightweight)
    priors_values: list[float] = [
        popularity_prior(doc.get("clicks")) + recency_prior(doc.get("age_days"))
        for doc in documents
    ]

    # Blend and return
    blended = blend_scores(bm25_values, semantic_values, priors_values, weights)
    id_score_pairs = [(str(doc["id"]), blended[i]) for i, doc in enumerate(documents)]
    return top_k(id_score_pairs, k)


def create_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[list[float]]:
    """
    Create embeddings for texts. Only loads sentence-transformers when called.

    Parameters
    ----------
    texts : list[str]
        Texts to embed
    model_name : str
        Sentence transformer model name

    Returns
    -------
    list[list[float]]
        Text embeddings
    """
    get_sentence_transformers()  # Ensure module is loaded
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings.tolist()


# Fast test function that loads instantly
def quick_test():
    """Test function that doesn't require any heavy imports."""
    test_docs = [
        {"id": "1", "tokens": ["python", "programming"], "clicks": 100},
        {"id": "2", "tokens": ["java", "programming"], "clicks": 50},
        {"id": "3", "tokens": ["machine", "learning"], "clicks": 200},
    ]

    results = rank_fast("python programming", test_docs, k=2, use_semantic=False)
    print(f"✅ Fast ranking test: {results}")
    return results


if __name__ == "__main__":
    print("Testing fast ranker (no heavy imports)...")
    quick_test()
    print("✅ Fast ranker loaded successfully!")
