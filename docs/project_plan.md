Project brief to give Cursor first: Canva Search/Ranker
Use this as the first instruction you paste into Cursor so it knows exactly what to build. It contains scope, features, algorithms, evaluation, constraints, and edge cases. No code.

Project name and goal

Build a single-file, interview-ready search and ranking module for Canva assets (templates, elements, photos, fonts).
Given a text query, return the top-k most relevant items with clear, working, testable, maintainable, and performant code.
Primary user journeys

Search for a design intent: “wedding invitation minimalist”, “blue resume”, “YouTube thumbnail neon”.
Filter-like constraints (lightweight): style tags, aspect ratio, content safety.
Expect diverse, non-duplicative results near the top.
Inputs and data model (minimal, with optional fields)

Query: a string.
Documents (items): at minimum include:
id: string identifier.
tokens: lowercase tokens from title/tags/description.
emb (optional): float vector embedding for semantic similarity, same length across docs if present.
clicks or saves: non-negative integer for popularity prior.
age_days: non-negative integer for recency prior.
Optional fields (for later enhancements, not required for baseline): type (template/photo/font), aspect_ratio, language, creator_quality, safe_flag, color/style tags.
Outputs

A stable, deterministic list of the top-k items, each with a final score (descending). If scores tie, break by id ascending.
Core features used for ranking (baseline vs nice-to-have)

Baseline features:
Lexical relevance: BM25 over tokens.
Semantic similarity: cosine(query_embedding, doc_embedding) when embeddings are available; otherwise 0.
Priors: popularity (log1p(clicks or saves)) and recency (exp decay on age_days).
Nice-to-have features (if time permits):
Aspect ratio match (when query hints at use case like “YouTube thumbnail”).
Language match or synonym expansion (resume vs CV).
Quality signals (creator_quality), and safe_search filtering.
Diversity and dedup via MMR to avoid near-identical results.
Algorithms and pipeline (what to implement)

Tokenization: lowercase, whitespace split, filter empties.
IDF computation: document-frequency-based IDF for BM25 on the toy corpus.
BM25 scoring: standard BM25 with k1 and b; avgdl precomputed per batch.
Semantic similarity: cosine similarity; query embedding is optional; keep a stub that returns 0 if not provided.
Priors:
Popularity: pop = log1p(clicks or saves).
Recency: rec = exp(-age_days / 30).
Score normalization and blending:
Z-normalize each component per candidate set (BM25, semantic, priors).
Combine with linear weights: score = α·BM25_z + β·semantic_z + γ·priors_z. Use sensible defaults and keep configurable.
Candidate generation:
Minimal version: score all docs with BM25, add semantic if available; take top-k with a heap for efficiency.
Optional hybrid recall: union of top-N BM25 and top-N semantic before blending.
Re-ranking (optional enhancement):
MMR diversification with item-item similarity from embeddings. Apply a near-duplicate threshold to drop dupes.
Determinism:
Stable tie-breakers by id. Seed any randomness if introduced (e.g., sampling).
Evaluation and metrics (to implement with small asserts)

NDCG@K for graded relevance lists.
MRR for first relevant position.
Recall@K (optional) for completeness checks on tiny test sets.
Minimal sanity tests that confirm obvious matches win and metrics behave on hand-crafted examples.
Performance and complexity targets

Single-file, standard library only; pure functions with type hints and docstrings.
Complexity:
BM25 score per doc: O(|q|).
Scoring all docs: O(N·|q|).
Top-k selection: O(N log k) via heap; avoid full sort when N is large.
MMR re-rank (optional): worst-case O(k·N).
Precompute IDF and avgdl once; cache tokenized query and avoid repeated work.
Constraints and guardrails

No external dependencies, no network calls, no file I/O.
Handle edge cases gracefully and deterministically.
Keep functions small and composable; separate scoring, normalization, ranking, metrics, and reranking.
Edge cases to handle explicitly

Empty query or whitespace-only query.
Empty corpus; zero-length documents.
Missing embeddings; differing embedding dimensions should yield 0 semantic similarity.
Duplicate documents or near-duplicates (optional MMR/dedup).
k <= 0, k > N.
All-zero component variance in normalization (return zeros safely).
Extremely old or extremely popular items should not overpower topical relevance (keep priors as light modifiers).
Acceptance criteria (Definition of Done)

Clear problem restatement at top of file/README.
Baseline ranker implemented with BM25 + priors; semantic component optional but supported.
Z-normalization and weighted blending with stable tie-breakers.
Minimal, deterministic tests for metrics and ranker behavior, including edge cases.
Performance notes and complexity stated; heap-based top-k used.
Optional enhancement implemented if time remains (e.g., MMR for diversity).
Brief summary of trade-offs and next steps for production (inverted index, ANN, A/B testing).
What not to include (out of scope for the interview baseline)

Full inverted index or ANN service; use straightforward loops and heaps in-code.
Personalization, session modeling, or heavyweight model training.
External services, databases, or large datasets.
Prompt to give Cursor after pasting this brief

Create README.md with the above project brief verbatim.
Then propose a function-by-function design (names, responsibilities, inputs/outputs, docstring summaries) that satisfies this brief. Do not generate code yet; only the design and a tiny test plan per function.
After I approve the design, generate the single-file implementation with minimal asserts and follow the acceptance criteria.
After code generation, add a short “Summary, Complexity, Next Steps” section at the bottom of the file.
