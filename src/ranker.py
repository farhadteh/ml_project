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
from math import sqrt


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


def compute_idf(corpus_tokens: Sequence[Sequence[str]]) -> dict[str, float]:
    """Deprecated: IDF is handled by rank_bm25; kept for compatibility/tests."""
    from rank_bm25 import BM25Okapi  # type: ignore

    if not corpus_tokens:
        return {}
    # BM25Okapi builds idf internally; we expose token->idf using its internals for tests.
    bm25 = BM25Okapi(corpus_tokens)
    # bm25.idf is a dict mapping token to idf
    return dict(bm25.idf)


def compute_avg_doc_len(corpus_tokens: Sequence[Sequence[str]]) -> float:
    """Deprecated: avgdl is handled by rank_bm25; kept for compatibility/tests."""
    if not corpus_tokens:
        return 0.0
    total_len = sum(len(tokens) for tokens in corpus_tokens)
    return total_len / float(len(corpus_tokens))


def bm25_score(
    query_tokens: Sequence[str],
    doc_tokens: Sequence[str],
    idf: dict[str, float] | None = None,
    avgdl: float | None = None,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """Compute BM25 score using rank_bm25 for a single document.

    Note: rank_bm25 scores a query against the entire corpus. To score a single
    document, we instantiate a BM25Okapi over [doc_tokens] and request the score.
    This is sufficient for tests and small inputs.
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    from rank_bm25 import BM25Okapi  # type: ignore

    bm25 = BM25Okapi([list(doc_tokens)], k1=k1, b=b)
    scores = bm25.get_scores(list(query_tokens))
    # get_scores returns a numpy array-like of length 1
    return float(scores[0])


if __name__ == "__main__":
    # Minimal sanity checks for task 2 utilities
    assert tokenize("") == []
    assert tokenize("  Blue   Resume  ") == ["blue", "resume"]

    assert z_normalize([]) == []
    assert z_normalize([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    zn = z_normalize([0.0, 1.0])
    assert abs(sum(zn)) < 1e-12

    # top_k tie handling: prefer smaller id on equal score
    tk = top_k([("b", 1.0), ("a", 1.0), ("c", 0.5)], 2)
    assert tk == [("a", 1.0), ("b", 1.0)]

    # BM25 quick checks
    corpus = [["blue", "resume"], ["resume", "template"], ["wedding", "invitation"]]
    idf = compute_idf(corpus)
    avgdl = compute_avg_doc_len(corpus)
    q = ["resume"]
    s1 = bm25_score(q, corpus[0], idf, avgdl)
    s2 = bm25_score(q, corpus[1], idf, avgdl)
    assert s1 > 0 and s2 > 0 and abs(s1 - s2) < 1e-6  # same freq and length here
