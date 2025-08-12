"""Unit tests for the single-file search engine.

These tests focus on the early implementation steps (1–5):
- Tokenization using spaCy (step 2)
- Context creation with BM25 models and TF-IDF matrix (step 3)
- BM25 scoring with field boosts and query weighting (step 4)
- Facet and popularity boosts monotonicity (step 5)
"""

from __future__ import annotations

from search_engine import (
    CONFIG,
    DOCS,
    apply_facet_boosts,
    apply_popularity_bonus,
    calculate_bm25_scores,
    cosine_similarity_optimized,
    cosine_similarity_sparse,
    create_search_context,
    detect_facets,
    expand_with_synonyms,
    mmr_select,
    search,
    tokenize,
)


def test_tokenize_basic() -> None:
    """Tokenization should lowercase and keep only alphanumeric tokens."""
    assert tokenize("A4 Resume!!") == ["a4", "resume"]


def test_context_shapes() -> None:
    """Search context must expose BM25 models (per field) and a TF-IDF matrix aligned to docs."""
    ctx = create_search_context(DOCS, CONFIG)
    assert ctx["tfidf_matrix"].shape[0] == len(DOCS)
    assert set(ctx["bm25"].keys()) == {"title", "tags", "desc"}


def test_bm25_and_boosts_monotonic() -> None:
    """Facet and popularity boosts must never reduce a document's score.

    We compute BM25 scores for a query, then apply facet boosts and popularity bonus.
    Each stage should be monotonic non-decreasing per document.
    """
    ctx = create_search_context(DOCS, CONFIG)
    tokens = tokenize("minimal a4 resume")
    facets, remaining = detect_facets(tokens, CONFIG)
    q = expand_with_synonyms(remaining, CONFIG["synonyms"], CONFIG["syn_weight"])
    base = calculate_bm25_scores(q, ctx, CONFIG)
    boosted = apply_facet_boosts(base, facets, ctx, CONFIG)
    final_scores = apply_popularity_bonus(boosted, ctx["popularity"], CONFIG["pop_weight"])
    # All boosts should be non-decreasing per doc_id
    for did in base:
        assert boosted[did] >= base[did] - 1e-12
        assert final_scores[did] >= boosted[did] - 1e-12


def test_cosine_similarity_sparse() -> None:
    """Cosine similarity should return values in [0,1] and handle edge cases."""
    # Basic similarity test
    vec_a = {"term1": 1.0, "term2": 0.5}
    vec_b = {"term1": 0.8, "term3": 0.6}
    sim = cosine_similarity_sparse(vec_a, vec_b)
    assert 0.0 <= sim <= 1.0

    # Empty vectors should return 0
    assert cosine_similarity_sparse({}, {"term": 1.0}) == 0.0
    assert cosine_similarity_sparse({"term": 1.0}, {}) == 0.0

    # Identical vectors should return 1
    vec = {"term": 1.0}
    assert abs(cosine_similarity_sparse(vec, vec) - 1.0) < 1e-10


def test_cosine_similarity_optimized() -> None:
    """Optimized cosine similarity should match sparse version but be faster."""
    import math

    # Test vectors from the example
    query_vec = {"ai": 0.6, "engineer": 0.8}
    query_norm = math.sqrt(0.36 + 0.64)  # = 1.0

    # Ad A: {ai: 0.5}
    ad_a = {"ai": 0.5}
    ad_a_norm = 0.5
    sim_a = cosine_similarity_optimized(query_vec, query_norm, ad_a, ad_a_norm)
    expected_a = (0.6 * 0.5) / (1.0 * 0.5)  # = 0.6
    assert abs(sim_a - expected_a) < 1e-10

    # Ad B: {health: 0.7} - no overlap
    ad_b = {"health": 0.7}
    ad_b_norm = 0.7
    sim_b = cosine_similarity_optimized(query_vec, query_norm, ad_b, ad_b_norm)
    assert sim_b == 0.0  # No shared terms

    # Ad C: {ai: 0.2, engineer: 0.2}
    ad_c = {"ai": 0.2, "engineer": 0.2}
    ad_c_norm = math.sqrt(0.04 + 0.04)  # ≈ 0.283
    sim_c = cosine_similarity_optimized(query_vec, query_norm, ad_c, ad_c_norm)
    expected_c = (0.6 * 0.2 + 0.8 * 0.2) / (1.0 * ad_c_norm)  # ≈ 0.99
    assert abs(sim_c - expected_c) < 1e-10

    # Test edge cases
    assert cosine_similarity_optimized({}, 1.0, {"term": 1.0}, 1.0) == 0.0
    assert cosine_similarity_optimized({"term": 1.0}, 0.0, {"term": 1.0}, 1.0) == 0.0


def test_mmr_select() -> None:
    """MMR should select diverse results and respect k parameter."""
    ctx = create_search_context(DOCS, CONFIG)

    # Create test candidates with some scores
    candidates = [("d1", 5.0), ("d2", 4.0), ("d3", 3.0), ("d4", 2.0)]

    # Test basic functionality
    results = mmr_select(candidates, k=2, ctx=ctx, config=CONFIG)
    assert len(results) == 2
    assert all(isinstance(item, tuple) and len(item) == 2 for item in results)

    # Test k=0 edge case
    assert mmr_select(candidates, k=0, ctx=ctx, config=CONFIG) == []

    # Test k larger than candidates
    results = mmr_select(candidates, k=10, ctx=ctx, config=CONFIG)
    assert len(results) == len(candidates)


def test_search_orchestrator() -> None:
    """End-to-end search should integrate all pipeline components."""
    ctx = create_search_context(DOCS, CONFIG)

    # Basic keyword search
    results = search("resume", ctx, CONFIG, k=3)
    assert len(results) <= 3
    assert all(isinstance(item, tuple) and len(item) == 2 for item in results)

    # Should find documents with "resume"
    result_ids = {r[0] for r in results}
    assert "d1" in result_ids or "d6" in result_ids

    # Empty query should return by popularity
    empty_results = search("", ctx, CONFIG, k=2)
    assert len(empty_results) <= 2

    # k=0 should return empty
    assert search("test", ctx, CONFIG, k=0) == []

    # Deterministic results
    query = "minimal template"
    results1 = search(query, ctx, CONFIG, k=3)
    results2 = search(query, ctx, CONFIG, k=3)
    assert results1 == results2


def test_search_facet_boosting() -> None:
    """Search with facets should boost matching documents."""
    ctx = create_search_context(DOCS, CONFIG)

    # Query with matching facets should boost d1 (size=A4, style=minimal)
    results = search("A4 minimal template", ctx, CONFIG, k=5)
    result_ids = [r[0] for r in results]

    # d1 should rank highly due to multiple facet matches
    assert "d1" in result_ids[:3]  # Should be in top 3


def test_search_synonym_expansion() -> None:
    """Search should expand synonyms and find related documents."""
    ctx = create_search_context(DOCS, CONFIG)

    # "cv" should expand to "resume" and find resume documents
    cv_results = search("cv", ctx, CONFIG, k=3)
    resume_results = search("resume", ctx, CONFIG, k=3)

    cv_ids = {r[0] for r in cv_results}
    resume_ids = {r[0] for r in resume_results}

    # Should have some overlap due to synonym expansion
    assert len(cv_ids & resume_ids) > 0
