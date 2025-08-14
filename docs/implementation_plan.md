### Gender Diversity Re-ranking — Implementation Plan (aligned with `docs/project_plan.md`)

This plan describes a self-contained pipeline in one Python file `re_ranker.py` that takes a list of 100
search results with relevance scores and gender tags, then applies three distinct re-ranking strategies to
improve gender diversity in the top-10 results. It follows the repo rules and is structured as a concise checklist.

---

## 1) Scope, Inputs/Outputs, and Constraints

- **Goal**: Re-rank search results to improve gender diversity while maintaining relevance quality.
- **Inputs**:
  - `results: list[dict]` with keys: `item_id: str`, `relevance_score: float`, `gender: str`
  - Initial list of 100 items with imbalanced gender distribution in top-10
- **Outputs**: Three different top-10 lists from different re-ranking strategies with diversity metrics.
- **Constraints**: Single `.py` file, Python 3.12, standard library only (no external dependencies).
  Pure functions where possible; deterministic behavior; 60-minute development timeframe.

---

## 2) Mock Data Construction

- Implement generator inside `re_ranker.py`:
  - `generate_mock_data() -> list[dict]`: 100 dictionaries with keys `item_id`, `relevance_score`, `gender`.
  - Relevance scores decrease from 1.0 to 0.0 (sorted high to low initially).
  - Gender distribution creates clear imbalance in top-10: ~8 Male, ~2 Female items.
  - Use deterministic seed for reproducible results.

---

## 3) Re-ranking Strategies

- Implement three distinct approaches:
  - **Simple Interleaving**: `rerank_interleave(results: list[dict]) -> list[dict]`
    - Alternate between highest-relevance items from majority and minority gender groups
    - Guarantees diversity but may significantly impact top relevance scores
  - **Boosted Demotion**: `rerank_boosted_demotion(results: list[dict], alpha: float = 0.1) -> list[dict]`
    - Apply penalty: `new_score = original_relevance - alpha * demotion_factor`
    - Demote over-represented gender after threshold (>5 items in top-10)
    - Maintains stronger link to original relevance scores
  - **Proportional Re-ranking**: `rerank_proportional(results: list[dict]) -> list[dict]`
    - Take top 5 items from each gender group, merge and sort within groups
    - Provides fixed 50:50 distribution with controlled relevance ordering

---

## 4) Evaluation and Metrics

- **Diversity Scoring**
  - `evaluate_diversity_score(top_10_results: list[dict]) -> int`:
    - Count items from minority gender in the top-10 list
    - Target: achieve count ≥ 3 (improvement from baseline ~2)
- **Baseline Calculation**
  - Calculate diversity score for original top-10 (sorted by relevance_score desc)
  - Establish baseline for comparison across all three strategies

---

## 5) Orchestration and Example Run

- Add `if __name__ == "__main__":` to:
  1) Generate mock data with `generate_mock_data()`.
  2) Calculate and display baseline diversity score.
  3) Apply all three re-ranking strategies.
  4) Calculate diversity scores for each strategy.
  5) Print comparison table showing strategy names, diversity scores, and trade-offs summary.

---

## 6) Edge Cases

- Empty or insufficient data for a gender group → handle gracefully with warning messages.
- All items have same relevance score → stable tie-breaking by `item_id`.
- Extreme gender imbalance (e.g., 100% one gender) → strategies should still attempt improvement.
- Alpha parameter in boosted demotion → validate range [0.0, 1.0] with sensible defaults.

---

## 7) Testing (minimal, deterministic)

- Add simple asserts in a `run_tests()` helper:
  - All three strategies improve diversity score vs. baseline.
  - Interleaving strategy produces exactly alternating gender pattern in early results.
  - Boosted demotion with alpha=0 equals original ranking; alpha=1 heavily demotes majority gender.
  - Proportional strategy produces exactly 5:5 gender split in top-10.

Example test invocation:

```bash
python re_ranker.py  # includes test assertions
```

---

## 8) Performance Notes

- All re-ranking strategies operate on pre-sorted lists, complexity O(N) where N=100.
- Interleaving requires O(N) space for gender group separation.
- Boosted demotion modifies scores in-place, then sorts: O(N log N).
- Proportional strategy: O(N) for group separation + O(k log k) for within-group sorting.

---

## 9) Acceptance Criteria (Definition of Done)

- [ ] One file `re_ranker.py` containing: mock data generator, three re-ranking strategies, and evaluation.
- [ ] Strategies include: simple interleaving, boosted demotion with alpha parameter, and proportional re-ranking.
- [ ] All three strategies demonstrate improved diversity score (≥ 3) compared to baseline (~2).
- [ ] Clear output showing baseline and strategy comparisons with trade-offs summary.
- [ ] Lint/format/tests pass locally; follows repo rules for Python 3.12 and type hints.

---

## 10) Conformance to Repo Rules (quick checklist)

- [ ] Python 3.12; type hints; small, pure functions with meaningful names.
- [ ] Keep imports organized; run `uv run ruff .` and `uv run black .`.
- [ ] Deterministic behavior with stable tie-breaking by `item_id`.
- [ ] Minimal asserts cover happy path and key edge cases; concise test functions [[memory:6073336]].

---

## 11) Milestones and Sign-off

1. [ ] Scaffold `re_ranker.py` with mock data generator producing imbalanced 100-item dataset.
2. [ ] Implement simple interleaving and boosted demotion strategies with type hints.
3. [ ] Implement proportional re-ranking strategy and diversity evaluation function.
4. [ ] Add main orchestration with clear output comparison and trade-offs summary.
5. [ ] Add minimal tests; format/lint; ensure all checks green and 60-minute timeframe met.

For each milestone, ensure code is type-annotated, deterministic, formatted, linted, and tests pass.
