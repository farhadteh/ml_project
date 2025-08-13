### Single-File Search-Ranking Prototype — Implementation Plan (aligned with `docs/project_plan.md`)

This plan describes a self-contained pipeline in one Python file `search_engine.py` that builds mock data,
generates hybrid features (BM25, embedding similarity, popularity), trains an `XGBRanker`, and exposes a
`rank_templates()` function. It follows the repo rules and is structured as a concise checklist.

---

## 1) Scope, Inputs/Outputs, and Constraints

- **Goal**: Rank template-like assets for a user query using a hybrid (keyword + semantic + business) model.
- **Inputs**:
  - `query: str`
  - `template_db: list[dict]` with keys: `template_id: str`, `title: str`, `description: str`,
    `popularity_score: int`
  - `k: int` top results to return (default 10)
- **Outputs**: Top-k list of `(template_id, score)`, sorted by predicted relevance desc (stable tie by id).
- **Constraints**: Single `.py` file, Python 3.12. Use `rank-bm25`, `sentence-transformers`, and `xgboost`.
  Pure functions where possible; deterministic behavior; do not require external I/O beyond library usage.

---

## 2) Mock Data Construction

- Implement generators inside `search_engine.py`:
  - `make_template_db() -> list[dict]`: 10–15 dictionaries with the required keys.
  - `make_search_logs(template_db) -> list[dict]`: training rows with keys `query`, `template_id`, `relevance`.
    For each unique query, create 4–5 rows: one relevance 3 (perfect match), one 2 (good semantic match),
    and remaining 1 or 0 (poor matches).

---

## 3) Feature Generation

- Provide a single entry point:
  - `generate_features(search_logs, template_db) -> tuple[X, y, groups]`
- Implementation details:
  - Convert inputs to pandas DataFrames; left-join logs with templates so each row is a (query, template) pair.
  - **BM25 (keyword feature)**: Use `rank-bm25` to compute keyword relevance per (query, description/title).
    Store as `bm25_score`.
  - **Embeddings (semantic feature)**: Using `sentence-transformers`, encode unique queries and template
    descriptions; compute cosine similarity for each pair, stored as `embedding_similarity`.
  - **Popularity (business feature)**: Normalize `popularity_score` into [0, 1] as `popularity_norm`.
  - Return:
    - `X`: DataFrame with columns `[bm25_score, embedding_similarity, popularity_norm]`
    - `y`: Series of `relevance`
    - `groups`: array of group sizes per query (e.g., `[4, 5, ...]`) for ranking training

---

## 4) Model Training and Ranking

- **Training**
  - Build an `xgboost.DMatrix` with features `X`, labels `y`, and `group=groups`.
  - Train via `xgboost.train` with `objective='rank:ndcg'`.
- **Inference**
  - `rank_templates(query, model, template_db) -> list[tuple[str, float]]`:
    - Construct features for the new `query` against all templates using the identical pipeline
      (BM25, embedding similarity, popularity normalization).
    - Predict with the trained model; sort descending by score; return `(template_id, score)`.

---

## 5) Orchestration and Example Run

- Add `if __name__ == "__main__":` to:
  1) Build `template_db` and `search_logs`.
  2) Call `generate_features` to get `X, y, groups`.
  3) Train the model.
  4) Call `rank_templates("professional resume", model, template_db)`.
  5) Print the ranked template titles to verify relevance.

---

## 6) Edge Cases

- Empty `template_db` or `search_logs` → return empty results or raise a clear error during training.
- Queries with no lexical overlap but semantic match → embeddings should still yield signal.
- Stable tie-breaking by `template_id` when scores are equal.
- Clamp `k` to `[0, N]`.

---

## 7) Testing (minimal, deterministic)

- Add simple asserts in a `run_tests()` helper:
  - A perfect lexical/semantic match receives higher predictions than unrelated templates.
  - A good semantic match (with little lexical overlap) still scores meaningfully above non-matches.
  - Popularity provides a small but non-dominant nudge.

Example test invocation:

```bash
uv run pytest -q
```

---

## 8) Performance Notes

- BM25 computation per (query, template) is lightweight for small N.
- Embedding computation amortized by encoding unique queries and template descriptions once.
- Training time is small for the mock dataset; prediction is O(N) per query.

---

## 9) Acceptance Criteria (Definition of Done)

- [x] One file `search_engine.py` containing: mock data builders, `generate_features`, model training, and
  `rank_templates`.
- [x] Features include: BM25 score, embedding cosine similarity, and normalized popularity.
- [x] Model trained with `objective='rank:ndcg'` and group-wise ranking via `xgboost.DMatrix`.
- [x] Example query returns a reasonable ranking (e.g., "Modern CV Design" ranks high for "professional resume").
- [x] Lint/format/tests pass locally.

---

## 10) Conformance to Repo Rules (quick checklist)

- [x] Python 3.12; type hints; small, pure functions.
- [x] Keep imports organized; run `uv run ruff .` and `uv run black .`.
- [x] Deterministic behavior with stable tie-breaking.
- [x] Minimal asserts cover happy path and key edge cases.

---

## 11) Milestones and Sign-off

1. [x] Scaffold `search_engine.py` with mock data generators.
2. [x] Implement `generate_features` (BM25, embeddings, popularity) and compute `groups`.
3. [x] Train `XGBRanker` via `xgboost.train` using `DMatrix` with groups.
4. [x] Implement `rank_templates` and example run.
5. [x] Add minimal tests; format/lint; ensure all checks green.

For each milestone, ensure code is type-annotated, deterministic, formatted, linted, and tests pass.
