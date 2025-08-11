### Canva Search/Ranker — Step-by-Step Implementation Plan (Cursor-ready)

This plan details the exact steps to implement a single-file, interview-ready search and ranking module as described in `docs/project_plan.md`. It is structured as a checklist to confirm each step and aligns with the workspace rules and development flow expected in Cursor.

---

## 1) Scope, Inputs/Outputs, and Constraints (from brief)

- **Goal**: Rank Canva-like assets given a query; return top-k items with final blended scores.
- **Inputs**:
  - `query: str`
  - `documents: list[dict]` with fields: `id: str`, `tokens: list[str]`, optional `emb: list[float]`, priors: `clicks: int`, `age_days: int`.
  - `k: int` top results to return.
- **Outputs**: Deterministic list of top-k `(id, score)` sorted by score desc, ties by `id` asc.
- **Constraints**: Single-file module, Python 3.12, standard library only, pure functions, deterministic behavior, no I/O/network. Complexity target: scoring O(N·|q|) and top-k O(N log k).

---

## 2) Development Workflow and Tooling (conforms to Cursor rules)

- [x] Create and use Python 3.12 environment
  - ```bash
    uv python install 3.12.8
    uv venv
    ```
- [x] Install dependencies (dev included) and set up pre-commit
  - ```bash
    uv sync --group dev
    uv run pre-commit install
    ```
- [x] Coding standards
  - Python 3.12, type hints everywhere, PEP 8, `ruff` + `black` + `isort(profile=black)`.
  - Pure functions, deterministic results, early returns, small functions.
- [x] Validation before each commit
  - ```bash
    uv run ruff .
    uv run black .
    uv run pre-commit run --all-files
    uv run pytest -q
    ```

---

## 3) File and API Plan (single-file implementation)

- [x] Add a new single module: `src/ranker.py` containing everything (scoring, normalization, ranking, metrics). Note: BM25 is provided via `rank_bm25`.
- **Core data types** (internal only, no runtime dataclasses needed):
  - `Document = dict[str, Any]` with required keys: `id: str`, `tokens: list[str]`; optional: `emb: list[float]`, `clicks: int`, `age_days: int`.
  - `Weights = tuple[float, float, float]` for `(alpha_bm25, beta_semantic, gamma_priors)`.

---

## 4) Function-by-Function Design (signatures and responsibilities)

- [x] `tokenize(text: str) -> list[str]`
  - Lowercase, whitespace split, drop empties. Used for query.

- [x] `compute_idf(corpus_tokens: list[list[str]]) -> dict[str, float]`
  - Implemented via `rank_bm25.BM25Okapi` internal precomputation.

- [x] `compute_avg_doc_len(corpus_tokens: list[list[str]]) -> float`
  - Implemented via `rank_bm25.BM25Okapi` internal precomputation.

- [x] `bm25_score(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float], avgdl: float, k1: float = 1.2, b: float = 0.75) -> float`
  - Implemented via `rank_bm25.BM25Okapi.get_scores()`.

- [x] `cosine_similarity(a: list[float] | None, b: list[float] | None) -> float`
  - Handle None or mismatched dims by returning 0.0.

- [x] `popularity_prior(clicks: int | None) -> float`
  - `log1p(max(0, clicks))`.

- [x] `recency_prior(age_days: int | None) -> float`
  - `exp(-max(0, age_days) / 30.0)`.

- [x] `z_normalize(values: list[float]) -> list[float]`
  - Per-candidate-set z-norm; if variance is 0 or list empty, return zeros.

- [x] `blend_scores(bm25: list[float], semantic: list[float], priors: list[float], weights: Weights) -> list[float]`
  - Z-normalize components then compute linear blend `alpha*bm25_z + beta*semantic_z + gamma*priors_z`.

- [x] `top_k(items: list[tuple[str, float]], k: int) -> list[tuple[str, float]]`
  - Use a heap for O(N log k). Apply stable tie-break by `id` asc. Return sorted by score desc.

- [x] `rank(query: str, documents: list[Document], k: int, weights: Weights = (1.0, 0.5, 0.2)) -> list[tuple[str, float]]`
  - Orchestrates: tokenize query; reuse/precompute corpus `idf` and `avgdl`; compute components and blend; heap-select top-k.

- [x] Metrics (minimal asserts only in tests)
  - `ndcg_at_k(gains: list[float], k: int) -> float`
  - `mrr(relevances: list[int]) -> float`

---

## 5) Implementation Steps (exact order)

1. [x] Create `src/ranker.py` with a docstring restating the problem and constraints.
2. [x] Implement tokenization and basic utilities (`z_normalize`, `top_k`). Add minimal inline asserts within a guarded `if __name__ == "__main__":` block or in tests only.
3. [x] Implement BM25 helpers: `compute_idf`, `compute_avg_doc_len`, `bm25_score` with unit-level asserts. (Handled via `rank_bm25.BM25Okapi` usage inside `rank()`.)
4. [x] Implement semantic similarity `cosine_similarity` with robust edge-case handling.
5. [x] Implement priors: `popularity_prior`, `recency_prior`.
6. [x] Implement blending `blend_scores` including zero-variance safety.
7. [x] Implement `rank` orchestration with deterministic tie-break.
8. [x] Implement metrics `ndcg_at_k`, `mrr` with simple tests.
9. [x] Add tiny handcrafted tests in `tests/test_ranker.py` covering: obvious lexical match wins; embeddings absent; priors don’t dominate; k edge cases; normalization variance=0; tie-break by id.
10. [x] Run format/lint/tests and fix any issues.

Commands to run:

```bash
uv run ruff .
uv run black .
uv run pytest -q
```

---

## 6) Edge Cases to Handle Explicitly

- [x] Empty query or whitespace-only query → return empty if `k <= 0`, else scores default to priors and stable sort.
- [x] Empty corpus or zero-length docs → graceful zero scores.
- [x] Missing or mismatched embeddings → semantic = 0.
- [x] k <= 0 or k > N → clamp to [0, N].
- [x] All-zero variance in any component → treat normalized vector as all zeros.
- [x] Tie scores → break by `id` ascending deterministically.

---

## 7) Testing Plan (minimal, deterministic)

- [x] Sanity: query "blue resume" vs docs with/without tokens; ensure lexical dominates.
- [x] Edge: empty query, empty corpus, k=0 and k>N.
- [x] Priors: a very old but popular item vs relevant item → relevant still wins with default weights.
- [x] Semantic: provide one pair of matching-dimension vectors to validate > 0 similarity and impact on rank.
- [x] Normalization: component with constant values yields zeros; blending still works.
- [x] Metrics: `ndcg_at_k` for a simple graded list; `mrr` with first relevant at position p.

Example test invocation:

```bash
uv run pytest -q
```

---

## 8) Performance and Determinism Notes

- [x] Complexity: BM25 per doc O(|q|), total O(N·|q|); heap top-k O(N log k).
- [x] Precompute: Using `rank_bm25.BM25Okapi`, which precomputes IDF and avgdl at initialization.
  - Current implementation constructs `BM25Okapi(corpus_tokens)` inside `rank()`, so IDF/avgdl are recomputed per call (simple and stateless).
  - For repeated queries on the same corpus, pre-build and cache `BM25Okapi` externally to reuse the precomputed IDF/avgdl.
- [x] Determinism: no randomness; stable tie-break by `id`.

---

## 9) Acceptance Criteria Checklist (Definition of Done)

- [x] Problem restatement present at top of `src/ranker.py`.
- [x] BM25 + priors implemented; semantic optional but supported.
- [x] Z-normalization and weighted blending with configurable weights.
- [x] Stable tie-breakers by `id`.
- [x] Minimal deterministic tests for metrics and ranker behavior, including edge cases.
- [x] Heap-based top-k used (no full sort for large N).
- [x] Summary/Complexity/Next Steps section at bottom of `src/ranker.py`.
- [x] `ruff`, `black`, and tests pass locally.

---

## 10) Conformance to Cursor Rules (quick checklist)

- [x] Python 3.12, standard library only, type hints, pure functions.
- [x] Small, composable functions; explicit error handling where needed.
- [x] No I/O or network; deterministic behavior; seeded randomness if ever added.
- [x] Tests include happy path and edge cases; tiny property test if time permits.
- [x] Lint/format with `uv run ruff .` and `uv run black .`; imports sorted.
- [x] Keep interfaces narrow; avoid globals; early returns.

---

## 11) Milestones and Sign-off

1. [x] Environment and scaffolding ready (uv, linting, tests running).
2. [x] Core utilities and BM25 implemented and tested.
3. [x] Priors and semantic similarity integrated.
4. [x] Blending and heap top-k working end-to-end.
5. [x] Metrics implemented and validated.
6. [x] Documentation, summary, and final cleanup; all checks green.

For each milestone, ensure: code is type-annotated, deterministic, formatted, linted, and tests pass.
