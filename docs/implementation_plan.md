### Single-File Search Engine — Step-by-Step Implementation Plan (aligned with `docs/project_plan.md`)

This plan details the exact steps to implement a self-contained search engine prototype in one Python file `search_engine.py`, implementing BM25 keyword relevance, synonym expansion, facet-based boosting, a popularity signal, and MMR re-ranking. It follows Cursor workspace rules and is structured as a checklist.

---

## 1) Scope, Inputs/Outputs, and Constraints

- **Goal**: Rank creative assets by a user text query; demonstrate a modern, multi-stage ranking pipeline.
- **Inputs**:
  - `query: str`
  - `docs: list[dict]` with fields: `id: str`, `title: str`, `tags: list[str]`, `desc: str`, `popularity: int`, optional facets like `size: str`, `style: str`, `color: str`.
  - `k: int` results to return (default 10).
- **Outputs**: Deterministic top-k list of `(doc_id, score)` after MMR, sorted by score desc, ties by `id` asc.
- **Constraints**: Single `.py` file, Python 3.12, standard library only, pure functions (no classes), deterministic, no I/O/network. Target complexities: BM25 scoring O(N·|q|); MMR O(k·N) with cosine sims over sparse TF-IDF.

---

## 2) Configuration and Mock Data

- **File scaffold**: `search_engine.py` contains:
  - `CONFIG: dict` with:
    - `k1: float`, `b: float` per field or global; `field_boosts: dict = {"title": 2.0, "tags": 1.5, "desc": 1.0}`
    - `synonyms: dict[str, list[str]]` (e.g., `{"cv": ["resume"], "photo": ["image", "picture"]}`)
    - `syn_weight: float` (0 < syn_weight < 1) to prefer original terms
    - `facets: dict[str, dict]` with keys like `size`, `style`, `color` and allowed values
    - `facet_boost: float` large positive boost per matched facet value
    - `pop_weight: float` small multiplier for popularity normalization
    - `mmr_lambda: float` in [0,1] for relevance vs diversity trade-off
  - `DOCS: list[dict]` of 10–15 mock assets with fields described above.

---

## 3) Search Context (precompute once)

- `create_search_context(docs, config) -> dict` builds and returns:
  - `inv_index: dict[field][term] -> list[(doc_id, tf)]`
  - `doc_len: dict[field][doc_id] -> int`, `avgdl: dict[field] -> float`
  - `df: dict[field][term] -> int`, `idf: dict[field][term] -> float` (BM25-friendly)
  - `tfidf_vectors: dict[doc_id] -> dict[token, weight]` over `title+tags` for cosine sims (MMR)
  - `popularity: dict[doc_id] -> float` normalized into [0,1]
  - `facets_index: dict[facet_name][value] -> set[doc_id]` (for quick facet matching)

---

## 4) Functions and Responsibilities

- `tokenize(text: str) -> list[str]`
  - Lowercase, split on non-alphanumerics; drop empties.
- `expand_with_synonyms(tokens: list[str], synonyms: dict[str, list[str]], syn_weight: float) -> list[tuple[str, float]]`
  - Return weighted query terms: originals with weight 1.0, synonyms with `syn_weight`.
- `detect_facets(tokens: list[str], config: dict) -> tuple[dict[str, str], list[str]]`
  - Extract `(facet_name -> value)` found in tokens; return remaining non-facet tokens.
- `create_search_context(docs: list[dict], config: dict) -> dict`
  - Build inverted index, df/idf, doc lengths, avgdl, tf-idf vectors, popularity normalization, facets index.
- `calculate_bm25_scores(query_terms: list[tuple[str, float]], ctx: dict, config: dict) -> dict[str, float]`
  - BM25 over fields with `field_boosts`; incorporate per-term query weights.
- `apply_facet_boosts(scores: dict[str, float], facet_matches: dict[str, str], ctx: dict, config: dict) -> dict[str, float]`
  - Add `facet_boost` to docs matching all query facets; partial matches receive proportionate boosts.
- `apply_popularity_bonus(scores: dict[str, float], popularity: dict[str, float], pop_weight: float) -> dict[str, float]`
  - Add a small `pop_weight * popularity[doc]` to favor well-liked items.
- `cosine_similarity_sparse(a: dict[str, float], b: dict[str, float]) -> float`
  - Sparse-dict cosine; 0.0 if empty.
- `mmr_select(candidates: list[tuple[str, float]], k: int, ctx: dict, config: dict) -> list[tuple[str, float]]`
  - Standard MMR using `tfidf_vectors` and `mmr_lambda`; greedy selection until k.
- `search(query: str, ctx: dict, config: dict, k: int = 10) -> list[tuple[str, float]]`
  - Orchestrate: tokenize → facet detection → synonym expansion → BM25 → facet boosts → popularity → top candidates → MMR.
- `run_tests() -> None`
  - Minimal asserts for keyword match, synonym expansion, facet boosting, popularity nudge, and MMR diversity.

---

## 5) Implementation Steps (exact order)

1. [x] Create `search_engine.py` with `CONFIG` and `DOCS` scaffold from the brief.
2. [x] Implement `tokenize`, `expand_with_synonyms`, and `detect_facets` with unit-level asserts.
3. [x] Implement `create_search_context` (inverted index, df/idf, lengths, avgdl, tf-idf vectors, popularity normalization, facets index).
4. [x] Implement `calculate_bm25_scores` supporting per-field boosts and weighted query terms.
5. [x] Implement `apply_facet_boosts` and `apply_popularity_bonus`.
6. [x] Implement `cosine_similarity_sparse` and `mmr_select`.
7. [x] Implement `search` orchestrator and `run_tests` with at least 3–4 asserts covering core scenarios.
8. [x] Format/lint and run tests:

```bash
uv run ruff .
uv run black .
uv run pytest -q
```

---

## 6) Edge Cases to Handle Explicitly

- **Empty query**: return empty list if `k <= 0`; otherwise rely on popularity-only ordering (deterministic tie-break by `id`).
- **Empty corpus**: return empty results.
- **Unknown facets/values**: ignore silently; proceed with remaining tokens.
- **All OOV tokens**: produce zero BM25; only facet/popularity can move scores.
- **k bounds**: clamp to `[0, N]`.
- **Ties**: stable tie-break by `id` ascending.

---

## 7) Testing Plan (deterministic, minimal)

- **Keyword**: a doc with `title` or `tags` containing query tokens should win over non-matching docs.
- **Synonyms**: query "cv" should retrieve docs with "resume" via expansion; originals preferred if both present.
- **Facets**: query with `"A4 minimal"` should boost items with `size="A4"` and `style="minimal"`.
- **Popularity**: more popular but less relevant items get a small nudge, not overpowering BM25.
- **MMR**: ensure top-k are not near-duplicates by title/tags; diversity increases when `mmr_lambda` < 1.

Example test invocation:

```bash
uv run pytest -q
```

---

## 8) Performance and Determinism Notes

- **Complexity**: BM25 O(N·|q|); MMR selection O(k·N) with sparse cosine; overall acceptable for N≲1000 in interview settings.
- **Precompute**: All per-corpus stats in `create_search_context`; query-time work is lightweight.
- **Determinism**: No randomness; explicit tie-breaking by `id`.

---

## 9) Acceptance Criteria (Definition of Done)

- [x] Single file `search_engine.py` with `CONFIG` and `DOCS` mock data.
- [x] BM25 with per-field boosts; synonym expansion with lower weight for synonyms.
- [x] Facet-based boosts and a small popularity bonus.
- [x] MMR post-ranking using cosine similarity over TF-IDF (title+tags).
- [x] Orchestrator `search()` and `run_tests()` with 3–4 asserts covering the above.
- [x] Deterministic results with stable tie-break; no external dependencies.
- [x] Lint/format/tests pass locally.

---

## 10) Conformance to Cursor Rules (quick checklist)

- [x] Python 3.12, standard library only, type hints, pure functions, early returns.
- [x] Small, composable functions; explicit error handling where appropriate.
- [x] No I/O or network; deterministic behavior.
- [x] Tests include happy path and edge cases.
- [x] Lint/format with `uv run ruff .` and `uv run black .`; imports sorted.

---

## 11) Milestones and Sign-off

1. [x] Scaffold `search_engine.py` with config and mock data.
2. [x] Indexing and context creation complete; unit checks pass.
3. [x] BM25 scoring with field boosts verified.
4. [x] Synonyms, facets, and popularity integrated and validated.
5. [x] MMR implemented; end-to-end tests pass.
6. [x] Cleanup, documentation, and final formatting; all checks green.

For each milestone, ensure: code is type-annotated, deterministic, formatted, linted, and tests pass.
