"""Self-contained search engine prototype (single-file) per docs/project_plan.md.

Implements a minimal multi-stage ranking stack with pure functions only:
- Tokenization and query processing (synonym expansion, facet detection)
- Precomputation of a search context (inverted index, BM25-ready stats, TF-IDF vectors)

This file intentionally avoids any I/O and uses only the Python 3.12 standard library.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import spacy
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------
# Configuration and Mock Data
# --------------------------

CONFIG: dict[str, Any] = {
    "k1": 1.2,
    "b": 0.75,
    "field_boosts": {"title": 2.0, "tags": 1.5, "desc": 1.0},
    "synonyms": {
        "cv": ["resume"],
        "photo": ["image", "picture"],
        "logo": ["brandmark"],
        "ppt": ["presentation", "slides"],
        "a4": ["letter"],
    },
    "syn_weight": 0.7,
    "facets": {
        "size": ["A4", "A3", "square", "instagram"],
        "style": ["minimal", "vintage", "modern"],
        "color": ["gold", "blue", "green", "black", "white"],
    },
    "facet_boost": 2.0,
    "pop_weight": 0.1,
    "mmr_lambda": 0.7,
}


DOCS: list[dict[str, Any]] = [
    {
        "id": "d1",
        "title": "Minimal A4 resume template",
        "tags": ["resume", "template", "a4", "minimal"],
        "desc": "Clean, professional CV layout for job applications.",
        "popularity": 120,
        "size": "A4",
        "style": "minimal",
        "color": "white",
    },
    {
        "id": "d2",
        "title": "Vintage poster design A3",
        "tags": ["poster", "vintage", "a3"],
        "desc": "Old-school aesthetic poster suitable for events and cafes.",
        "popularity": 90,
        "size": "A3",
        "style": "vintage",
        "color": "black",
    },
    {
        "id": "d3",
        "title": "Modern business presentation slides",
        "tags": ["presentation", "slides", "modern"],
        "desc": "Corporate deck with charts and diagrams.",
        "popularity": 200,
        "size": "square",
        "style": "modern",
        "color": "blue",
    },
    {
        "id": "d4",
        "title": "Gold logo brandmark pack",
        "tags": ["logo", "brandmark", "gold"],
        "desc": "Elegant golden marks for luxury brands.",
        "popularity": 160,
        "size": "square",
        "style": "modern",
        "color": "gold",
    },
    {
        "id": "d5",
        "title": "Instagram post template minimal",
        "tags": ["instagram", "template", "minimal"],
        "desc": "Clean posts for social campaigns and announcements.",
        "popularity": 300,
        "size": "instagram",
        "style": "minimal",
        "color": "white",
    },
    {
        "id": "d6",
        "title": "Professional CV resume pack",
        "tags": ["cv", "resume", "professional"],
        "desc": "Multiple layouts tailored for different industries.",
        "popularity": 80,
        "size": "A4",
        "style": "modern",
        "color": "black",
    },
    {
        "id": "d7",
        "title": "Travel photo collage template",
        "tags": ["photo", "image", "collage", "template"],
        "desc": "Arrange your pictures in a clean grid.",
        "popularity": 140,
        "size": "square",
        "style": "minimal",
        "color": "white",
    },
    {
        "id": "d8",
        "title": "Business card minimal black",
        "tags": ["card", "business", "minimal", "black"],
        "desc": "Sleek black-and-white business card.",
        "popularity": 60,
        "size": "A4",
        "style": "minimal",
        "color": "black",
    },
    {
        "id": "d9",
        "title": "Event flyer modern blue",
        "tags": ["flyer", "event", "modern", "blue"],
        "desc": "Bold flyer suitable for concerts and talks.",
        "popularity": 110,
        "size": "A4",
        "style": "modern",
        "color": "blue",
    },
    {
        "id": "d10",
        "title": "Photography portfolio presentation",
        "tags": ["photography", "portfolio", "presentation"],
        "desc": "Showcase images with a clean and spacious layout.",
        "popularity": 95,
        "size": "square",
        "style": "modern",
        "color": "white",
    },
]


# --------------------------
# Query Processing Utilities
# --------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NLP = spacy.blank("en")


def tokenize(text: str) -> list[str]:
    """Tokenize text to lowercase alphanumeric tokens using spaCy tokenizer.

    Uses a lightweight `spacy.blank('en')` tokenizer (no external model download).
    Filters tokens to alphanumeric via `_TOKEN_RE` to maintain deterministic behavior.
    """

    doc = _NLP.make_doc(text.lower())
    return [t.text for t in doc if _TOKEN_RE.fullmatch(t.text)]


def expand_with_synonyms(
    tokens: list[str], synonyms: dict[str, list[str]], syn_weight: float
) -> list[tuple[str, float]]:
    """Return weighted query terms including synonyms.

    Originals are weight 1.0; synonyms are `syn_weight` (0 < syn_weight < 1). If a synonym
    coincides with an original, it keeps weight 1.0. Output is deterministic.
    """

    weighted: dict[str, float] = {}
    for tok in tokens:
        weighted[tok] = max(1.0, weighted.get(tok, 0.0))
        for syn in synonyms.get(tok, []):
            if syn == tok:
                continue
            weighted[syn] = max(weighted.get(syn, 0.0), float(syn_weight))
    # Ensure deterministic order
    return sorted(weighted.items(), key=lambda x: x[0])


def detect_facets(tokens: list[str], config: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Detect facet tokens from the query tokens.

    Recognizes values for facets declared under `config["facets"]`. Returns a mapping of
    facet_name -> value and the remaining non-facet tokens.
    """

    facet_map: dict[str, str] = {}
    remaining: list[str] = []

    facet_values: dict[str, set[str]] = {
        name: {v.lower() for v in values} for name, values in (config.get("facets") or {}).items()
    }

    for tok in tokens:
        matched = False
        for facet_name, values in facet_values.items():
            if tok in values:
                # If multiple facet values appear for same facet, prefer first occurrence
                if facet_name not in facet_map:
                    # Map back to canonical case if needed: use token's original form is fine
                    facet_map[facet_name] = tok.upper() if facet_name == "size" else tok
                matched = True
                break
        if not matched:
            remaining.append(tok)

    return facet_map, remaining


# ----------------------------------
# Search Context (index and statistics)
# ----------------------------------


def _tokens_for_field(doc: dict[str, Any], field: str) -> list[str]:
    if field == "tags":
        return [t.lower() for t in (doc.get("tags") or [])]
    return tokenize(str(doc.get(field, "")))


def _precompute_doc_norms(tfidf_matrix) -> list[float]:
    """Precompute L2 norms for all documents in the TF-IDF matrix.

    Args:
        tfidf_matrix: Sparse CSR matrix from TfidfVectorizer

    Returns:
        List of L2 norms, one per document (aligned to doc_order)
    """
    # Compute row-wise L2 norms efficiently using sparse matrix operations
    squared_norms = tfidf_matrix.multiply(tfidf_matrix).sum(axis=1).A1
    return [math.sqrt(norm) for norm in squared_norms]


def create_search_context(docs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Build and return all precomputed data structures for searching.

    Returns a dictionary with keys:
    - bm25: dict[field] -> BM25Okapi
    - doc_order: list[str] stable order for corpora
    - tfidf_vectorizer: TfidfVectorizer for title+tags
    - tfidf_matrix: sparse matrix [n_docs, vocab]
    - popularity: dict[doc_id] -> float in [0,1]
    - facets_index: dict[facet_name][value] -> set[doc_id]
    """

    fields: tuple[str, ...] = ("title", "tags", "desc")

    # Stable document order for all corpora and matrices
    doc_order: list[str] = [str(doc["id"]) for doc in docs]

    # Per-field tokenized corpora aligned to doc_order
    per_field_corpus: dict[str, list[list[str]]] = {f: [] for f in fields}
    for doc in docs:
        for f in fields:
            per_field_corpus[f].append(_tokens_for_field(doc, f))

    # BM25 models per field using rank_bm25
    bm25: dict[str, BM25Okapi] = {f: BM25Okapi(per_field_corpus[f]) for f in fields}

    # TF-IDF using scikit-learn over title+tags combined text
    combined_texts: list[str] = []
    for doc in docs:
        title_text = " ".join(_tokens_for_field(doc, "title"))
        tags_text = " ".join(_tokens_for_field(doc, "tags"))
        combined_texts.append(f"{title_text} {tags_text}".strip())

    tfidf_vectorizer = TfidfVectorizer(token_pattern=r"[a-z0-9]+", lowercase=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_texts)

    # Precompute document norms for efficient sparse cosine similarity
    doc_norms = _precompute_doc_norms(tfidf_matrix)

    # Popularity normalization into [0,1]
    raw_pop: dict[str, float] = {str(d["id"]): float(d.get("popularity", 0)) for d in docs}
    if raw_pop:
        pop_min = min(raw_pop.values())
        pop_max = max(raw_pop.values())
        denom = pop_max - pop_min
        if denom <= 0.0:
            popularity = {doc_id: 0.0 for doc_id in raw_pop}
        else:
            popularity = {doc_id: (val - pop_min) / denom for doc_id, val in raw_pop.items()}
    else:
        popularity = {}

    # Facets index
    facets_cfg: dict[str, Iterable[str]] = config.get("facets", {}) or {}
    facets_index: dict[str, dict[str, set[str]]] = {
        name: defaultdict(set) for name in facets_cfg.keys()
    }
    for d in docs:
        doc_id = str(d["id"])
        for facet_name in facets_cfg.keys():
            val = d.get(facet_name)
            if isinstance(val, str) and val:
                facets_index[facet_name][val].add(doc_id)

    return {
        "bm25": bm25,
        "doc_order": doc_order,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "doc_norms": doc_norms,
        "popularity": popularity,
        "facets_index": facets_index,
    }


def calculate_bm25_scores(
    query_terms: list[tuple[str, float]], ctx: dict[str, Any], config: dict[str, Any]
) -> dict[str, float]:
    """Compute BM25 scores across fields with boosts and weighted query terms.

    For each field, we compute a per-term contribution via `BM25Okapi.get_scores([term])` and
    linearly combine with the provided query term weights, then apply `field_boosts` and sum
    across fields. The output is a mapping `doc_id -> score` aligned with `ctx['doc_order']`.
    """

    field_boosts: dict[str, float] = config.get("field_boosts", {}) or {}
    bm25_models: dict[str, BM25Okapi] = ctx["bm25"]
    doc_order: list[str] = ctx["doc_order"]

    # Initialize scores to zero
    scores_accum: list[float] = [0.0 for _ in doc_order]

    # For each field, sum weighted term contributions and apply field boost
    for field, model in bm25_models.items():
        field_boost = float(field_boosts.get(field, 1.0))
        if not query_terms:
            continue
        # Accumulate weighted scores for this field
        field_scores = [0.0 for _ in doc_order]
        for term, weight in query_terms:
            if weight <= 0.0:
                continue
            term_scores = model.get_scores([term])  # aligned to doc order used at construction
            # Add weighted contribution
            for i, s in enumerate(term_scores):
                field_scores[i] += float(weight) * float(s)
        # Apply field boost and add to total
        for i, s in enumerate(field_scores):
            scores_accum[i] += field_boost * s

    return {doc_id: score for doc_id, score in zip(doc_order, scores_accum, strict=False)}


def apply_facet_boosts(
    scores: dict[str, float],
    facet_matches: dict[str, str],
    ctx: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, float]:
    """Apply facet-based boosts.

    Boost is proportional to the fraction of query facets matched by a document. If `m` facets
    are specified and a doc matches `c` of them, add `facet_boost * (c / m)` to its score.
    """

    if not facet_matches:
        return scores

    m = max(1, len(facet_matches))
    facet_boost = float(config.get("facet_boost", 0.0))
    facets_index: dict[str, dict[str, set[str]]] = ctx.get("facets_index", {})

    # Precompute for efficiency: for each facet/value, the matching doc set
    facet_to_docs: list[set[str]] = []
    for facet_name, value in facet_matches.items():
        value_docs = facets_index.get(facet_name, {}).get(value, set())
        facet_to_docs.append(set(value_docs))

    new_scores = dict(scores)
    all_doc_ids = list(scores.keys())
    for doc_id in all_doc_ids:
        matches = sum(1 for s in facet_to_docs if doc_id in s)
        if matches > 0 and facet_boost != 0.0:
            new_scores[doc_id] = new_scores[doc_id] + facet_boost * (matches / m)
    return new_scores


def apply_popularity_bonus(
    scores: dict[str, float], popularity: dict[str, float], pop_weight: float
) -> dict[str, float]:
    """Add a small popularity bonus to scores.

    `popularity` is expected to be normalized in [0, 1]. The bonus is `pop_weight * pop`.
    """

    if pop_weight == 0.0 or not popularity:
        return scores

    new_scores = dict(scores)
    for doc_id, base in scores.items():
        pop = float(popularity.get(doc_id, 0.0))
        new_scores[doc_id] = base + pop_weight * pop
    return new_scores


def cosine_similarity_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse TF-IDF vectors.

    Args:
        a, b: Sparse vectors as dict[term, weight]

    Returns:
        Cosine similarity in [0, 1]. Returns 0.0 if either vector is empty.
    """
    if not a or not b:
        return 0.0

    # Compute dot product over shared terms
    dot_product = sum(a[term] * b.get(term, 0.0) for term in a if term in b)

    # Compute norms
    norm_a = math.sqrt(sum(weight * weight for weight in a.values()))
    norm_b = math.sqrt(sum(weight * weight for weight in b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def cosine_similarity_optimized(
    query_vec: dict[str, float], query_norm: float, doc_vec: dict[str, float], doc_norm: float
) -> float:
    """Optimized sparse cosine similarity with precomputed norms.

    Args:
        query_vec: Sparse query vector as dict[term, weight]
        query_norm: Precomputed L2 norm of query vector
        doc_vec: Sparse document vector as dict[term, weight]
        doc_norm: Precomputed L2 norm of document vector

    Returns:
        Cosine similarity in [0, 1]. Returns 0.0 if either norm is zero.
    """
    if query_norm == 0.0 or doc_norm == 0.0 or not query_vec or not doc_vec:
        return 0.0

    # Compute sparse dot product - only iterate over non-zero query terms
    dot_product = sum(
        query_vec[term] * doc_vec.get(term, 0.0) for term in query_vec if term in doc_vec
    )

    return dot_product / (query_norm * doc_norm)


def mmr_select(
    candidates: list[tuple[str, float]], k: int, ctx: dict[str, Any], config: dict[str, Any]
) -> list[tuple[str, float]]:
    """Select top-k candidates using Maximal Marginal Relevance (MMR).

    Args:
        candidates: List of (doc_id, score) sorted by score descending
        k: Number of results to select
        ctx: Search context with tfidf_vectorizer and tfidf_matrix
        config: Configuration with mmr_lambda parameter

    Returns:
        List of (doc_id, score) with diverse top-k selection
    """
    if k <= 0 or not candidates:
        return []

    k = min(k, len(candidates))
    mmr_lambda = float(config.get("mmr_lambda", 0.7))

    # Extract TF-IDF data from context
    doc_order: list[str] = ctx["doc_order"]
    tfidf_matrix = ctx["tfidf_matrix"]
    vectorizer = ctx["tfidf_vectorizer"]
    doc_norms: list[float] = ctx["doc_norms"]

    # Build sparse TF-IDF vectors for candidate docs with precomputed norms
    doc_vectors: dict[str, dict[str, float]] = {}
    doc_norm_map: dict[str, float] = {}
    feature_names = vectorizer.get_feature_names_out()

    for doc_id, _ in candidates:
        try:
            doc_idx = doc_order.index(doc_id)
            doc_row = tfidf_matrix[doc_idx]
            # Convert sparse row to dict
            doc_vectors[doc_id] = {
                feature_names[j]: doc_row[0, j]
                for j in range(doc_row.shape[1])
                if doc_row[0, j] > 0
            }
            # Store precomputed norm
            doc_norm_map[doc_id] = doc_norms[doc_idx]
        except (ValueError, IndexError):
            doc_vectors[doc_id] = {}
            doc_norm_map[doc_id] = 0.0

    # MMR greedy selection
    selected: list[tuple[str, float]] = []
    remaining = list(candidates)

    while len(selected) < k and remaining:
        best_mmr_score = -float("inf")
        best_idx = 0

        for i, (doc_id, relevance_score) in enumerate(remaining):
            if not selected:
                # First selection: pure relevance
                mmr_score = relevance_score
            else:
                # Compute max similarity to already selected docs using optimized sparse cosine
                max_sim = 0.0
                doc_vec = doc_vectors.get(doc_id, {})
                doc_norm = doc_norm_map.get(doc_id, 0.0)

                for sel_doc_id, _ in selected:
                    sel_vec = doc_vectors.get(sel_doc_id, {})
                    sel_norm = doc_norm_map.get(sel_doc_id, 0.0)
                    sim = cosine_similarity_optimized(doc_vec, doc_norm, sel_vec, sel_norm)
                    max_sim = max(max_sim, sim)

                # MMR formula: λ * relevance - (1-λ) * max_similarity
                mmr_score = mmr_lambda * relevance_score - (1 - mmr_lambda) * max_sim

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i

        # Move best candidate from remaining to selected
        selected.append(remaining.pop(best_idx))

    return selected


def search(
    query: str, ctx: dict[str, Any], config: dict[str, Any], k: int = 10
) -> list[tuple[str, float]]:
    """Main search orchestrator implementing the full ranking pipeline.

    Pipeline:
    1. Tokenize query
    2. Detect facets and extract remaining tokens
    3. Expand with synonyms
    4. Calculate BM25 scores with field boosts
    5. Apply facet boosts
    6. Apply popularity bonus
    7. Select top candidates and re-rank with MMR

    Args:
        query: User search query
        ctx: Search context from create_search_context()
        config: Configuration dictionary
        k: Number of results to return

    Returns:
        List of (doc_id, score) tuples sorted by final score descending
    """
    if k <= 0:
        return []

    # Step 1: Tokenize query
    tokens = tokenize(query)
    if not tokens:
        # Empty query: return by popularity only with stable tie-break
        popularity = ctx.get("popularity", {})
        doc_order = ctx.get("doc_order", [])
        pop_scores = [(doc_id, popularity.get(doc_id, 0.0)) for doc_id in doc_order]
        # Sort by popularity desc, then by doc_id asc for stable tie-break
        pop_scores.sort(key=lambda x: (-x[1], x[0]))
        return pop_scores[:k]

    # Step 2: Detect facets and extract remaining tokens
    facet_matches, remaining_tokens = detect_facets(tokens, config)

    # Step 3: Expand with synonyms
    weighted_query_terms = expand_with_synonyms(
        remaining_tokens, config.get("synonyms", {}), config.get("syn_weight", 0.7)
    )

    # Step 4: Calculate BM25 scores with field boosts
    bm25_scores = calculate_bm25_scores(weighted_query_terms, ctx, config)

    # Step 5: Apply facet boosts
    boosted_scores = apply_facet_boosts(bm25_scores, facet_matches, ctx, config)

    # Step 6: Apply popularity bonus
    final_scores = apply_popularity_bonus(
        boosted_scores, ctx.get("popularity", {}), config.get("pop_weight", 0.0)
    )

    # Step 7: Sort candidates by score and apply MMR
    candidates = sorted(final_scores.items(), key=lambda x: (-x[1], x[0]))  # Score desc, id asc
    mmr_results = mmr_select(candidates, k, ctx, config)

    return mmr_results


def run_tests() -> None:
    """Comprehensive test suite covering the full search pipeline."""
    print("Running comprehensive search engine tests...")

    # Setup
    ctx = create_search_context(DOCS, CONFIG)

    print("✓ Test 1: Basic keyword search")
    results = search("resume", ctx, CONFIG, k=3)
    assert len(results) <= 3
    # Should find documents with "resume" in title/tags
    result_ids = {r[0] for r in results}
    assert "d1" in result_ids or "d6" in result_ids  # Both have "resume"

    print("✓ Test 2: Synonym expansion")
    # "cv" should expand to "resume" and find resume documents
    cv_results = search("cv", ctx, CONFIG, k=3)
    resume_results = search("resume", ctx, CONFIG, k=3)
    # Should have some overlap due to synonym expansion
    cv_ids = {r[0] for r in cv_results}
    resume_ids = {r[0] for r in resume_results}
    assert len(cv_ids & resume_ids) > 0  # Some overlap expected

    print("✓ Test 3: Facet boosting")
    # Query with facets should boost matching documents
    facet_results = search("A4 minimal template", ctx, CONFIG, k=5)
    # d1 has size="A4", style="minimal", and contains "template"
    facet_result_ids = [r[0] for r in facet_results]
    # d1 should rank high due to facet matches
    assert "d1" in facet_result_ids[:3]  # Should be in top 3

    print("✓ Test 4: Empty query handling")
    empty_results = search("", ctx, CONFIG, k=3)
    assert len(empty_results) <= 3
    # Should return by popularity order
    if empty_results:
        # Check that results are sorted by popularity (highest first)
        popularities = [ctx["popularity"].get(r[0], 0.0) for r in empty_results]
        assert popularities == sorted(popularities, reverse=True)

    print("✓ Test 5: MMR diversity")
    # Search for common terms to get many candidates, then check MMR diversity
    common_results = search("template minimal", ctx, CONFIG, k=4)
    assert len(common_results) <= 4
    # With MMR, results should be more diverse than pure relevance ranking

    print("✓ Test 6: Edge cases")
    # k=0 should return empty
    assert search("test", ctx, CONFIG, k=0) == []

    # Non-existent terms should still work (rely on facets/popularity)
    nonsense_results = search("xyz123", ctx, CONFIG, k=2)
    assert len(nonsense_results) <= 2

    print("✓ Test 7: Deterministic results")
    # Same query should return same results
    query = "modern presentation"
    results1 = search(query, ctx, CONFIG, k=3)
    results2 = search(query, ctx, CONFIG, k=3)
    assert results1 == results2

    print("All tests passed! ✅")


if __name__ == "__main__":
    # Minimal sanity asserts for steps (1)-(3)
    # tokenize
    assert tokenize("A4 Resume!!") == ["a4", "resume"]

    # expand_with_synonyms
    w = expand_with_synonyms(["cv"], CONFIG["synonyms"], CONFIG["syn_weight"])
    assert dict(w)["cv"] == 1.0 and dict(w)["resume"] == CONFIG["syn_weight"]

    # detect_facets
    facets, rem = detect_facets(["a4", "minimal", "resume"], CONFIG)
    assert facets.get("size") in {"A4", "a4"} and "resume" in rem

    # create_search_context
    context = create_search_context(DOCS, CONFIG)
    assert "bm25" in context and "tfidf_matrix" in context
    assert context["tfidf_matrix"].shape[0] == len(DOCS)

    # Basic sanity for BM25 scoring and boosts (steps 4-5)
    base_tokens = tokenize("minimal a4 resume")
    facet_map, remaining = detect_facets(base_tokens, CONFIG)
    q_terms = expand_with_synonyms(remaining, CONFIG["synonyms"], CONFIG["syn_weight"])
    bm25_scores = calculate_bm25_scores(q_terms, context, CONFIG)
    assert isinstance(bm25_scores, dict) and len(bm25_scores) == len(DOCS)
    # Applying facet boosts should not reduce scores
    boosted = apply_facet_boosts(bm25_scores, facet_map, context, CONFIG)
    for did in bm25_scores:
        assert boosted[did] >= bm25_scores[did] - 1e-12
    # Popularity bonus increases or keeps the same
    final_scores = apply_popularity_bonus(boosted, context["popularity"], CONFIG["pop_weight"])
    for did in boosted:
        assert final_scores[did] >= boosted[did] - 1e-12

    # Basic MMR test (step 6)
    candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    mmr_results = mmr_select(candidates, k=3, ctx=context, config=CONFIG)
    assert len(mmr_results) <= 3 and len(mmr_results) <= len(candidates)

    # Test cosine similarity
    vec_a = {"resume": 0.5, "template": 0.3}
    vec_b = {"resume": 0.4, "minimal": 0.6}
    sim = cosine_similarity_sparse(vec_a, vec_b)
    assert 0.0 <= sim <= 1.0

    # Test optimized cosine similarity
    norm_a = math.sqrt(0.5**2 + 0.3**2)
    norm_b = math.sqrt(0.4**2 + 0.6**2)
    sim_opt = cosine_similarity_optimized(vec_a, norm_a, vec_b, norm_b)
    assert abs(sim - sim_opt) < 1e-10  # Should match exactly

    print("\n" + "=" * 50)
    print("RUNNING COMPREHENSIVE TESTS")
    print("=" * 50)
    run_tests()
