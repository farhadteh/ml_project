from __future__ import annotations

from ranker import cosine_similarity, mrr, ndcg_at_k, rank, z_normalize


def test_z_normalize_constant_returns_zeros() -> None:
    values = [1.0, 1.0, 1.0, 1.0]
    normalized = z_normalize(values)
    assert normalized == [0.0, 0.0, 0.0, 0.0]


def test_metrics_ndcg_and_mrr() -> None:
    # Perfect ranking should yield NDCG@3 == 1.0
    gains = [3.0, 2.0, 1.0]
    assert abs(ndcg_at_k(gains, 3) - 1.0) < 1e-12

    # MRR: first relevant at position 2 (0-indexed) -> 1/(2+1) = 1/3
    assert abs(mrr([0, 0, 1, 0]) - (1.0 / 3.0)) < 1e-12


def test_rank_empty_corpus_returns_empty() -> None:
    assert rank("query", [], k=5) == []


def test_cosine_similarity_basic() -> None:
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-12
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-12
    assert cosine_similarity(None, [1.0]) == 0.0
    assert cosine_similarity([1.0], None) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0]) == 0.0


def test_rank_handles_zero_length_doc_tokens() -> None:
    docs = [
        {"id": "a", "tokens": []},
        {"id": "b", "tokens": ["x"]},
    ]
    results = rank("x", docs, k=2, weights=(1.0, 0.0, 0.0))
    assert [doc_id for doc_id, _ in results] == ["b", "a"]
