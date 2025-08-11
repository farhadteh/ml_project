from __future__ import annotations

from math import isclose

from rank_bm25 import BM25Okapi


def test_bm25_basic_symmetry() -> None:
    """BM25 scoring is symmetric for bags with the same tokens in different order."""
    corpus = [["blue", "resume"], ["resume", "blue"]]
    bm25 = BM25Okapi(corpus)
    q = ["blue", "resume"]
    scores = bm25.get_scores(q)
    assert isclose(float(scores[0]), float(scores[1]), rel_tol=1e-9, abs_tol=1e-12)


def test_bm25_handles_empty_query_and_docs() -> None:
    """BM25 returns zero for empty query and for empty docs, guarding edge cases."""
    # Empty query should yield zero BM25 scores
    corpus = [["x"], []]
    bm25 = BM25Okapi(corpus)
    scores_empty_query = bm25.get_scores([])
    assert float(scores_empty_query[0]) == 0.0
    assert float(scores_empty_query[1]) == 0.0

    # Empty document should yield zero score even for non-empty query
    scores_with_query = bm25.get_scores(["x"])
    assert float(scores_with_query[1]) == 0.0
