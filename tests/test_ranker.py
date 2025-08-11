from __future__ import annotations

from typing import Any

from data.sample_data import get_sample_documents
from ranker import rank


def test_rank_basic_lexical_top2_contains_resume_docs() -> None:
    """Lexical matching with a multi-term query returns the two resume-related docs.

    Ensures BM25 over tokens surfaces documents containing both "resume" and "template".
    """
    docs = get_sample_documents()
    results = rank("resume template", docs, k=2)
    top_ids = [doc_id for doc_id, _ in results]
    assert set(top_ids) == {"doc_001", "doc_004"}


def test_rank_empty_query_uses_priors() -> None:
    """Empty query falls back to priors (popularity + recency).

    The doc with the highest prior should be ranked first deterministically.
    """
    docs = get_sample_documents()
    results = rank("", docs, k=1)
    # Highest prior (log1p(clicks) + exp(-age/30)) should be doc_001
    assert results[0][0] == "doc_001"


def test_rank_tie_break_on_id_when_scores_equal() -> None:
    """Stable tie-break: when scores are equal, prefer lexicographically smaller id."""
    docs: list[dict[str, Any]] = [
        {"id": "a", "tokens": ["x"], "clicks": 0, "age_days": 0},
        {"id": "b", "tokens": ["x"], "clicks": 0, "age_days": 0},
    ]
    results = rank("x", docs, k=2, weights=(1.0, 0.0, 0.0))
    assert results == [("a", results[0][1]), ("b", results[1][1])]


def test_rank_lexical_beats_priors_with_weights() -> None:
    """When BM25 is weighted higher than priors, a lexical hit outranks popular irrelevance."""
    docs: list[dict[str, Any]] = [
        {"id": "rel", "tokens": ["resume"], "clicks": 0, "age_days": 10},
        {"id": "popular_irrel", "tokens": ["wedding"], "clicks": 10, "age_days": 10},
    ]
    # Increase BM25 weight, reduce priors so lexical match wins deterministically
    results = rank("resume", docs, k=2, weights=(1.0, 0.0, 0.1))
    assert results[0][0] == "rel"


def test_rank_k_edge_cases() -> None:
    """k bounds are respected: k<=0 returns empty; large k caps at corpus size."""
    docs = get_sample_documents()
    assert rank("resume", docs, k=0) == []
    res = rank("resume", docs, k=999)
    assert len(res) == len(docs)
