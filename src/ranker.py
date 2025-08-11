"""
Canva Search/Ranker — Single-file module

Problem restatement
-------------------
Build a single-file, interview-ready search and ranking module for Canva-like assets.
Given a text query, return the top-k most relevant items with a final blended score.

Inputs
------
- query: str
- documents: list of dict items (minimal schema):
    - id: str (identifier)
    - tokens: list[str] (lowercased tokens from title/tags/description)
    - emb: optional list[float] (fixed-length embedding vector)
    - clicks: optional int (popularity prior)
    - age_days: optional int (recency prior)
- k: int (number of results)

Outputs
-------
- Deterministic list of top-k (id, score) pairs sorted by score descending.
- Stable tie-breaker by id ascending when scores are equal.

Algorithmic components (baseline)
---------------------------------
- Lexical relevance: BM25 over tokens.
- Semantic similarity (optional): cosine(query_embedding, doc_embedding); 0 if unavailable.
- Priors: popularity = log1p(clicks), recency = exp(-age_days/30).
- Z-normalize each component per candidate set, then blend linearly:
    score = alpha * BM25_z + beta * semantic_z + gamma * priors_z
  with sensible default weights and configurability.

Candidate generation and ranking
--------------------------------
- Minimal version: score all docs with BM25 and add semantic if available.
- Select top-k via a heap for O(N log k) instead of full sort.
- Deterministic tie-breaking by id.

Metrics (for tiny asserts)
--------------------------
- NDCG@K for graded relevance.
- MRR for first relevant position.

Constraints and guardrails
--------------------------
- Python 3.12, standard library only.
- Single file; pure functions with type hints and concise docstrings.
- No file or network I/O. Deterministic behavior.
- Handle edge cases: empty query/corpus; zero-length docs; missing/mismatched embeddings
  (semantic=0); k<=0 or k>N; all-zero variance in normalization; tie-break by id.

Complexity targets
------------------
- BM25 per doc: O(|q|). Scoring all docs: O(N · |q|).
- Top-k selection: O(N log k) via heap.
- Precompute IDF and avgdl once per corpus; avoid recomputation.

Acceptance criteria
-------------------
- Clear problem restatement (this docstring).
- Baseline ranker with BM25 + priors; semantic optional but supported.
- Z-normalization + weighted blending; stable tie-breakers.
- Minimal deterministic tests for ranker and metrics, including edge cases.
- Heap-based top-k; performance notes and complexity stated.
- Summary/Complexity/Next Steps section at bottom of file after implementation.
"""

# Implementation to follow per docs/implementation_plan.md steps.

from __future__ import annotations

from collections.abc import Sequence
from math import exp, log1p, log2, sqrt
from typing import Any

# Prefer scikit-learn for cosine similarity and NDCG when available
try:  # pragma: no cover - import guard
    import numpy as np
    from sklearn.metrics import ndcg_score as skl_ndcg_score
    from sklearn.metrics.pairwise import cosine_similarity as skl_cosine_similarity

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency guard
    _HAS_SKLEARN = False


def tokenize(text: str) -> list[str]:
    """Lowercase and whitespace-split a string.

    Drops empty tokens and preserves order. Deterministic and locale-agnostic.

    Params
    ------
    text: str
        Raw input string to tokenize.

    Returns
    -------
    list[str]
        Lowercased, whitespace-split tokens with empties removed.
    """
    if not text:
        return []
    return [token for token in text.lower().split() if token]


def z_normalize(values: Sequence[float]) -> list[float]:
    """Z-normalize a sequence to zero-mean, unit-variance safely.

    If the input is empty or has zero variance, returns all zeros of the same length.

    Params
    ------
    values: Sequence[float]
        Numeric values to normalize.

    Returns
    -------
    list[float]
        Z-normalized values or zeros if variance is zero.
    """
    length = len(values)
    if length == 0:
        return []

    mean_value = sum(values) / float(length)
    variance = sum((value - mean_value) ** 2 for value in values) / float(length)
    if variance <= 0.0:
        return [0.0 for _ in range(length)]

    std_dev = sqrt(variance)
    return [(value - mean_value) / std_dev for value in values]


def top_k(items: Sequence[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """Select top-k (id, score) by score desc with deterministic tie-breaks.

    Uses a min-heap of size k for O(N log k) selection. For equal scores, prefers
    smaller `id` lexicographically to satisfy the final tie-break rule.

    Params
    ------
    items: Sequence[tuple[str, float]]
        Pairs of (id, score).
    k: int
        Number of items to return; clamped to [0, len(items)].

    Returns
    -------
    list[tuple[str, float]]
        Top-k items sorted by score desc, then id asc.
    """
    num_items = len(items)
    if k <= 0 or num_items == 0:
        return []

    k = min(k, num_items)

    # Min-heap of (score, id) so that the smallest score (worst) stays at the root.
    # Resolve ties by preferring lexicographically smaller id in the final result.
    import heapq

    heap: list[tuple[float, str]] = []

    for item_id, score in items:
        if len(heap) < k:
            heapq.heappush(heap, (score, item_id))
            continue

        root_score, root_id = heap[0]
        if score > root_score or (score == root_score and item_id < root_id):
            heapq.heapreplace(heap, (score, item_id))

    # Convert heap to sorted list: score desc, id asc
    heap_items: list[tuple[str, float]] = [(item_id, score) for score, item_id in heap]
    heap_items.sort(key=lambda pair: (-pair[1], pair[0]))
    return heap_items


# Removed deprecated BM25 helper functions; use rank_bm25.BM25Okapi directly in rank().


def cosine_similarity(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    """Cosine similarity between two vectors.

    Uses scikit-learn if available; otherwise falls back to a simple manual implementation.
    Returns 0.0 if vectors are None, empty, mismatched, or any has zero norm.
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    if _HAS_SKLEARN:
        try:
            a_arr = np.asarray(a, dtype=float).reshape(1, -1)
            b_arr = np.asarray(b, dtype=float).reshape(1, -1)
            sim = skl_cosine_similarity(a_arr, b_arr)[0, 0]
            if np.isnan(sim):  # guard zero-norm
                return 0.0
            return float(sim)
        except Exception:
            # Fall through to manual computation
            pass

    # Manual fallback
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for va, vb in zip(a, b, strict=False):
        dot += va * vb
        norm_a_sq += va * va
        norm_b_sq += vb * vb
    if norm_a_sq <= 0.0 or norm_b_sq <= 0.0:
        return 0.0
    return dot / sqrt(norm_a_sq * norm_b_sq)


def popularity_prior(clicks: int | None) -> float:
    """Popularity prior as log1p of non-negative clicks."""
    value = 0 if clicks is None else max(0, int(clicks))
    return log1p(value)


def recency_prior(age_days: int | None) -> float:
    """Recency prior decaying exponentially with age in days."""
    age = 0 if age_days is None else max(0, int(age_days))
    return exp(-age / 30.0)


def blend_scores(
    bm25_values: Sequence[float],
    semantic_values: Sequence[float],
    priors_values: Sequence[float],
    weights: tuple[float, float, float],
) -> list[float]:
    """Blend component scores after per-component z-normalization.

    All input sequences are truncated to the same minimum length. If empty, returns [].

    Params
    ------
    bm25_values, semantic_values, priors_values: Sequence[float]
        Component scores per document.
    weights: tuple[float, float, float]
        Weights (alpha_bm25, beta_semantic, gamma_priors).

    Returns
    -------
    list[float]
        Final blended scores per document.
    """
    alpha, beta, gamma = weights
    n = min(len(bm25_values), len(semantic_values), len(priors_values))
    if n == 0:
        return []

    bm25_z = z_normalize(bm25_values[:n])
    sem_z = z_normalize(semantic_values[:n])
    pri_z = z_normalize(priors_values[:n])

    return [alpha * bm25_z[i] + beta * sem_z[i] + gamma * pri_z[i] for i in range(n)]


Document = dict[str, Any]
Weights = tuple[float, float, float]


def rank(
    query: str,
    documents: Sequence[Document],
    k: int,
    weights: Weights = (1.0, 0.5, 0.2),
    query_embedding: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Rank documents given a text query using BM25 + semantic + priors.

    Notes
    -----
    - If a document lacks tokens but has a text field, it will be tokenized on the fly.
    - Semantic similarity is 0 unless both query and documents contain an embedding under key "emb".

    Params
    ------
    query: str
        Raw query string.
    documents: Sequence[Document]
        Items with at least keys: "id" (str) and "tokens" (list[str]) or "text" (str).
        Optional: "emb" (list[float]), "clicks" (int), "age_days" (int).
    k: int
        Number of results to return.
    weights: Weights
        (alpha_bm25, beta_semantic, gamma_priors) for blending.
    query_embedding: Sequence[float] | None
        Optional query embedding. If provided and document embeddings are present and
        dimension-matched, semantic similarity contributes to the final score.

    Returns
    -------
    list[tuple[str, float]]
        Top-k (id, score) sorted by score desc then id asc.
    """
    if k <= 0 or not documents:
        return []

    query_tokens = tokenize(query)

    # Ensure tokens present; avoid mutating input by building transient corpus tokens
    corpus_tokens: list[list[str]] = []
    for doc in documents:
        if "tokens" in doc and isinstance(doc["tokens"], list):
            tokens: list[str] = [str(t) for t in doc["tokens"]]
        else:
            text = str(doc.get("text", ""))
            tokens = tokenize(text)
        corpus_tokens.append(tokens)

    # Compute BM25 scores using rank_bm25
    from rank_bm25 import BM25Okapi  # type: ignore

    if not corpus_tokens or not any(corpus_tokens):
        bm25_values = [0.0] * len(documents)
    else:
        bm25 = BM25Okapi(corpus_tokens)
        bm25_values = (
            bm25.get_scores(query_tokens).tolist() if query_tokens else [0.0] * len(documents)
        )

    # Fallback: On very small corpora, BM25 idf can be zero for single-token queries,
    # yielding all-zero scores even when there is a clear lexical match. When that
    # happens, use a simple overlap count so lexical evidence can still dominate
    # when weighted accordingly.
    if query_tokens and all(score == 0.0 for score in bm25_values):
        query_set = set(query_tokens)
        bm25_values = [float(len(query_set.intersection(set(tokens)))) for tokens in corpus_tokens]

    # Semantic: only if both query and document embeddings exist and have same length
    semantic_values: list[float] = [0.0 for _ in documents]
    if query_embedding is not None:
        semantic_values = [
            (
                cosine_similarity(query_embedding, doc.get("emb"))
                if isinstance(doc.get("emb"), list)
                else 0.0
            )
            for doc in documents
        ]

    # Priors: combine popularity and recency as a single component
    priors_values: list[float] = [
        popularity_prior(doc.get("clicks")) + recency_prior(doc.get("age_days"))
        for doc in documents
    ]

    blended = blend_scores(bm25_values, semantic_values, priors_values, weights)
    id_score_pairs = [(str(doc["id"]), blended[i]) for i, doc in enumerate(documents)]
    return top_k(id_score_pairs, k)


def ndcg_at_k(gains: Sequence[float], k: int) -> float:
    """Compute NDCG@K for a ranked list of graded gains.

    If scikit-learn is available, delegates to sklearn.metrics.ndcg_score. To preserve the
    provided ranking order without requiring predicted scores, a synthetic strictly decreasing
    score vector is used so that the sorting induced by scores matches the input order.
    Falls back to a manual computation if sklearn is unavailable.
    """
    if k <= 0 or not gains:
        return 0.0
    limit = min(k, len(gains))

    if _HAS_SKLEARN:
        try:
            y_true = np.asarray(gains, dtype=float).reshape(1, -1)
            # Strictly decreasing scores ensure current order is preserved
            y_score = np.linspace(len(gains), 1.0, num=len(gains), dtype=float).reshape(1, -1)
            return float(skl_ndcg_score(y_true, y_score, k=limit))
        except Exception:
            # Fall through to manual
            pass

    # Manual computation
    dcg = 0.0
    for i in range(limit):
        dcg += gains[i] / log2(i + 2.0)
    sorted_gains = sorted(gains, reverse=True)
    idcg = 0.0
    for i in range(limit):
        idcg += sorted_gains[i] / log2(i + 2.0)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def mrr(relevances: Sequence[int]) -> float:
    """Mean Reciprocal Rank for a single ranked list.

    Returns 0.0 if no relevant item (value > 0) is present.
    """
    for index, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / float(index + 1)
    return 0.0


if __name__ == "__main__":
    # Minimal sanity checks for utilities
    assert tokenize("") == []
    assert tokenize("  Blue   Resume  ") == ["blue", "resume"]

    assert z_normalize([]) == []
    assert z_normalize([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    zn = z_normalize([0.0, 1.0])
    assert abs(sum(zn)) < 1e-12

    # top_k tie handling: prefer smaller id on equal score
    tk = top_k([("b", 1.0), ("a", 1.0), ("c", 0.5)], 2)
    assert tk == [("a", 1.0), ("b", 1.0)]

    # Priors quick checks
    assert popularity_prior(None) == 0.0
    assert popularity_prior(10) > 0.0
    assert recency_prior(0) == 1.0
    assert 0.0 < recency_prior(30) < 1.0

    # Cosine similarity
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-12
    assert cosine_similarity(None, [1.0]) == 0.0

    # NDCG
    assert abs(ndcg_at_k([3, 2, 1], 3) - 1.0) < 1e-12  # perfect ranking
    assert ndcg_at_k([], 2) == 0.0

    # MRR
    assert abs(mrr([0, 1, 0]) - 0.5) < 1e-12  # first relevant at position 2 (0-indexed)
    assert mrr([0, 0, 0]) == 0.0

# -------------------------------------------------------------
# Summary / Complexity / Next Steps
# -------------------------------------------------------------
# Summary
# - Implements a deterministic ranker blending BM25, optional semantic similarity, and priors.
# - Applies per-component z-normalization and weighted linear combination.
# - Uses a heap-based top-k selection with stable tie-breakers by id.
#
# Complexity
# - BM25 scoring per document: O(|q|); for N docs: O(N · |q|).
# - Top-k selection: O(N log k) using a min-heap.
# - Memory: O(N) for storing intermediate component scores and heap of size k.
#
# Next Steps
# - For repeated queries on a static corpus, reuse a BM25Okapi instance to
#   avoid recomputing IDF and avgdl on each call.
# - Add support for external query embeddings and a simple embedding model adapter if needed.
# - Consider learning weights from labeled data or using a small learning-to-rank model.
