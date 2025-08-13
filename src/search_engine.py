"""
Single-file search-ranking prototype with hybrid features and XGBRanker.

This module implements a complete pipeline:
1. Mock data generation (templates and search logs)
2. Feature generation (BM25, embeddings, popularity)
3. XGBRanker training and ranking function
"""

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def make_template_db() -> list[dict[str, Any]]:
    """Generate mock template database with realistic design assets."""
    templates = [
        {
            "template_id": "t001",
            "title": "Modern Business Card",
            "description": "Clean and professional business card design with minimalist layout",
            "popularity_score": 4500,
        },
        {
            "template_id": "t002",
            "title": "Creative Resume Template",
            "description": "Modern CV design with creative typography and clean sections",
            "popularity_score": 3200,
        },
        {
            "template_id": "t003",
            "title": "Wedding Invitation Card",
            "description": "Elegant wedding invitation with floral decorations and script fonts",
            "popularity_score": 2800,
        },
        {
            "template_id": "t004",
            "title": "Corporate Presentation Slides",
            "description": "Professional PowerPoint template for business presentations",
            "popularity_score": 4100,
        },
        {
            "template_id": "t005",
            "title": "Birthday Party Invite",
            "description": "Fun and colorful birthday invitation template for kids parties",
            "popularity_score": 1900,
        },
        {
            "template_id": "t006",
            "title": "Restaurant Menu Design",
            "description": "Modern restaurant menu layout with food photography sections",
            "popularity_score": 2200,
        },
        {
            "template_id": "t007",
            "title": "Social Media Post Template",
            "description": "Instagram post template with trendy graphics and text overlays",
            "popularity_score": 3800,
        },
        {
            "template_id": "t008",
            "title": "Professional Portfolio",
            "description": "Clean portfolio website template for creative professionals",
            "popularity_score": 3500,
        },
        {
            "template_id": "t009",
            "title": "Event Flyer Design",
            "description": "Eye-catching flyer template for concerts and entertainment events",
            "popularity_score": 2600,
        },
        {
            "template_id": "t010",
            "title": "Annual Report Layout",
            "description": "Corporate annual report template with charts and data visualization",
            "popularity_score": 1500,
        },
        {
            "template_id": "t011",
            "title": "Logo Design Template",
            "description": "Minimalist logo template with geometric shapes and modern typography",
            "popularity_score": 4200,
        },
        {
            "template_id": "t012",
            "title": "Newsletter Template",
            "description": "Email newsletter design with header, content sections and footer",
            "popularity_score": 2400,
        },
    ]
    return templates


def make_search_logs(template_db: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate realistic search interaction logs for training."""
    search_logs = []

    # Query: "professional resume"
    search_logs.extend(
        [
            {
                "query": "professional resume",
                "template_id": "t002",
                "relevance": 3,
            },  # Perfect match
            {
                "query": "professional resume",
                "template_id": "t008",
                "relevance": 2,
            },  # Good semantic match (portfolio)
            {
                "query": "professional resume",
                "template_id": "t001",
                "relevance": 1,
            },  # Some relevance (business)
            {
                "query": "professional resume",
                "template_id": "t005",
                "relevance": 0,
            },  # No relevance (birthday)
            {
                "query": "professional resume",
                "template_id": "t009",
                "relevance": 0,
            },  # No relevance (flyer)
        ]
    )

    # Query: "birthday invite"
    search_logs.extend(
        [
            {"query": "birthday invite", "template_id": "t005", "relevance": 3},  # Perfect match
            {
                "query": "birthday invite",
                "template_id": "t003",
                "relevance": 2,
            },  # Good match (invitation)
            {
                "query": "birthday invite",
                "template_id": "t009",
                "relevance": 1,
            },  # Some relevance (event)
            {"query": "birthday invite", "template_id": "t002", "relevance": 0},  # No relevance
            {"query": "birthday invite", "template_id": "t010", "relevance": 0},  # No relevance
        ]
    )

    # Query: "business card"
    search_logs.extend(
        [
            {"query": "business card", "template_id": "t001", "relevance": 3},  # Perfect match
            {
                "query": "business card",
                "template_id": "t004",
                "relevance": 2,
            },  # Good match (corporate)
            {
                "query": "business card",
                "template_id": "t008",
                "relevance": 1,
            },  # Some relevance (professional)
            {"query": "business card", "template_id": "t006", "relevance": 0},  # No relevance
            {"query": "business card", "template_id": "t007", "relevance": 0},  # No relevance
        ]
    )

    # Query: "social media post"
    search_logs.extend(
        [
            {"query": "social media post", "template_id": "t007", "relevance": 3},  # Perfect match
            {
                "query": "social media post",
                "template_id": "t009",
                "relevance": 2,
            },  # Good match (visual content)
            {
                "query": "social media post",
                "template_id": "t012",
                "relevance": 1,
            },  # Some relevance (content)
            {"query": "social media post", "template_id": "t003", "relevance": 0},  # No relevance
            {"query": "social media post", "template_id": "t010", "relevance": 0},  # No relevance
        ]
    )

    # Query: "wedding invitation"
    search_logs.extend(
        [
            {"query": "wedding invitation", "template_id": "t003", "relevance": 3},  # Perfect match
            {
                "query": "wedding invitation",
                "template_id": "t005",
                "relevance": 2,
            },  # Good match (invitation)
            {
                "query": "wedding invitation",
                "template_id": "t009",
                "relevance": 1,
            },  # Some relevance (event)
            {"query": "wedding invitation", "template_id": "t001", "relevance": 0},  # No relevance
            {"query": "wedding invitation", "template_id": "t004", "relevance": 0},  # No relevance
        ]
    )

    return search_logs


def generate_features(
    search_logs: list[dict[str, Any]], template_db: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    """
    Generate hybrid features for ranking: BM25, embedding similarity, and popularity.

    Returns:
        X: Feature matrix with columns [bm25_score, embedding_similarity, popularity_norm]
        y: Relevance scores
        groups: Group sizes for each unique query (required for XGBRanker)
    """
    # Convert to DataFrames
    logs_df = pd.DataFrame(search_logs)
    templates_df = pd.DataFrame(template_db)

    # Merge to get full (query, template) pairs
    merged_df = logs_df.merge(templates_df, on="template_id", how="left")

    # 1. BM25 Feature: keyword relevance
    unique_queries = merged_df["query"].unique()

    # Prepare corpus (combine title + description for each template)
    corpus_texts = []
    template_id_to_idx = {}
    for idx, template in enumerate(template_db):
        corpus_texts.append(f"{template['title']} {template['description']}")
        template_id_to_idx[template["template_id"]] = idx

    # Tokenize corpus for BM25
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # Calculate BM25 scores for each (query, template) pair
    bm25_scores = []
    for _, row in merged_df.iterrows():
        query_tokens = row["query"].lower().split()
        template_idx = template_id_to_idx[row["template_id"]]
        score = bm25.get_scores(query_tokens)[template_idx]
        bm25_scores.append(score)

    merged_df["bm25_score"] = bm25_scores

    # 2. Embedding Feature: semantic similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode unique queries and template descriptions
    query_embeddings = model.encode(unique_queries.tolist())
    template_texts = [f"{t['title']} {t['description']}" for t in template_db]
    template_embeddings = model.encode(template_texts)

    # Calculate cosine similarity for each (query, template) pair
    embedding_similarities = []
    for _, row in merged_df.iterrows():
        query_idx = np.where(unique_queries == row["query"])[0][0]
        template_idx = template_id_to_idx[row["template_id"]]

        query_emb = query_embeddings[query_idx].reshape(1, -1)
        template_emb = template_embeddings[template_idx].reshape(1, -1)
        similarity = cosine_similarity(query_emb, template_emb)[0][0]
        embedding_similarities.append(similarity)

    merged_df["embedding_similarity"] = embedding_similarities

    # 3. Popularity Feature: normalize to [0, 1]
    max_popularity = max(t["popularity_score"] for t in template_db)
    min_popularity = min(t["popularity_score"] for t in template_db)

    popularity_norm = []
    for _, row in merged_df.iterrows():
        if max_popularity == min_popularity:
            norm_score = 0.5  # All equal, use middle value
        else:
            norm_score = (row["popularity_score"] - min_popularity) / (
                max_popularity - min_popularity
            )
        popularity_norm.append(norm_score)

    merged_df["popularity_norm"] = popularity_norm

    # Prepare outputs
    X = merged_df[["bm25_score", "embedding_similarity", "popularity_norm"]]
    y = merged_df["relevance"]

    # Calculate group sizes for XGBRanker
    groups = merged_df.groupby("query").size().tolist()

    return X, y, groups


def train_ranking_model(X: pd.DataFrame, y: pd.Series, groups: list[int]) -> xgb.Booster:
    """
    Train XGBRanker model for template ranking.

    Args:
        X: Feature matrix with columns [bm25_score, embedding_similarity, popularity_norm]
        y: Relevance scores (0-3)
        groups: Group sizes for each unique query

    Returns:
        Trained XGBoost model
    """
    # Create DMatrix with group information for ranking
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(groups)

    # XGBoost parameters for ranking
    params = {
        "objective": "rank:ndcg",
        "eta": 0.1,  # Learning rate
        "max_depth": 6,  # Tree depth
        "eval_metric": "ndcg@5",  # Evaluation metric
        "verbosity": 0,  # Suppress verbose output
    }

    # Train the model
    num_rounds = 100
    model = xgb.train(params, dtrain, num_rounds)

    return model


def rank_templates(
    query: str, model: xgb.Booster, template_db: list[dict[str, Any]], k: int = 10
) -> list[tuple[str, float]]:
    """
    Rank templates for a given query using the trained model.

    Args:
        query: Search query string
        model: Trained XGBoost ranking model
        template_db: List of template dictionaries
        k: Number of top results to return

    Returns:
        List of (template_id, score) tuples, sorted by score descending
    """
    if not template_db:
        return []

    # Clamp k to valid range
    k = max(0, min(k, len(template_db)))
    if k == 0:
        return []

    # Note: template_db is already a list of dicts, no need to convert to DataFrame

    # 1. Calculate BM25 scores
    corpus_texts = []
    for template in template_db:
        corpus_texts.append(f"{template['title']} {template['description']}")

    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)

    # 2. Calculate embedding similarities
    model_emb = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode query and template texts
    query_embedding = model_emb.encode([query])
    template_texts = [f"{t['title']} {t['description']}" for t in template_db]
    template_embeddings = model_emb.encode(template_texts)

    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, template_embeddings)[0]

    # 3. Normalize popularity scores
    popularity_scores = [t["popularity_score"] for t in template_db]
    max_popularity = max(popularity_scores)
    min_popularity = min(popularity_scores)

    if max_popularity == min_popularity:
        popularity_norm = [0.5] * len(template_db)  # All equal, use middle value
    else:
        popularity_norm = [
            (score - min_popularity) / (max_popularity - min_popularity)
            for score in popularity_scores
        ]

    # 4. Create feature matrix
    X_rank = pd.DataFrame(
        {
            "bm25_score": bm25_scores,
            "embedding_similarity": similarities,
            "popularity_norm": popularity_norm,
        }
    )

    # 5. Predict scores using the trained model
    dtest = xgb.DMatrix(X_rank)
    predicted_scores = model.predict(dtest)

    # 6. Create results with template_id and scores
    results = []
    for i, template in enumerate(template_db):
        results.append((template["template_id"], float(predicted_scores[i])))

    # 7. Sort by score descending, with stable tie-breaking by template_id
    results.sort(key=lambda x: (-x[1], x[0]))

    # 8. Return top-k results
    return results[:k]


def run_tests() -> None:
    """Basic tests for the complete ranking pipeline."""
    # Test mock data generation
    template_db = make_template_db()
    search_logs = make_search_logs(template_db)

    assert len(template_db) >= 10, "Should have at least 10 templates"
    assert len(search_logs) >= 20, "Should have multiple search interactions"
    assert all("template_id" in t for t in template_db), "All templates need template_id"
    assert all("relevance" in log for log in search_logs), "All logs need relevance scores"

    # Test feature generation
    X, y, groups = generate_features(search_logs, template_db)

    assert X.shape[1] == 3, "Should have 3 features: BM25, embedding, popularity"
    assert len(X) == len(y), "Feature matrix and labels should have same length"
    assert sum(groups) == len(X), "Groups should sum to total number of samples"
    assert X["bm25_score"].notna().all(), "BM25 scores should not be NaN"
    assert X["embedding_similarity"].notna().all(), "Embedding similarities should not be NaN"
    assert (X["popularity_norm"] >= 0).all() and (
        X["popularity_norm"] <= 1
    ).all(), "Popularity should be normalized [0,1]"

    # Test model training
    model = train_ranking_model(X, y, groups)
    assert model is not None, "Model should be trained successfully"

    # Test ranking functionality
    test_queries = ["professional resume", "birthday invite", "business card"]

    for query in test_queries:
        results = rank_templates(query, model, template_db, k=5)

        assert len(results) <= 5, f"Should return at most 5 results for '{query}'"
        assert all(isinstance(r[0], str) for r in results), "Template IDs should be strings"
        assert all(isinstance(r[1], float) for r in results), "Scores should be floats"

        # Check ordering (scores should be descending)
        scores = [r[1] for r in results]
        assert scores == sorted(
            scores, reverse=True
        ), f"Results should be sorted by score desc for '{query}'"

    # Test edge cases
    empty_results = rank_templates("", model, [], k=5)
    assert empty_results == [], "Empty template_db should return empty results"

    zero_k_results = rank_templates("test", model, template_db, k=0)
    assert zero_k_results == [], "k=0 should return empty results"

    # Test semantic matching: "cv" should rank "resume" highly
    cv_results = rank_templates("cv", model, template_db, k=3)
    cv_template_ids = [r[0] for r in cv_results]
    assert "t002" in cv_template_ids, "Resume template should rank highly for 'cv' query"

    print("✓ All tests passed!")


if __name__ == "__main__":
    # Run tests
    run_tests()

    # Example pipeline execution
    print("\n=== Mock Data Generation ===")
    template_db = make_template_db()
    search_logs = make_search_logs(template_db)

    print(f"Generated {len(template_db)} templates")
    print(f"Generated {len(search_logs)} search interactions")

    print("\n=== Feature Generation ===")
    X, y, groups = generate_features(search_logs, template_db)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Groups for ranking: {groups}")
    print("\nFeature statistics:")
    print(X.describe())

    print("\n=== Model Training ===")
    model = train_ranking_model(X, y, groups)
    print("✓ XGBRanker model trained successfully!")

    print("\n=== Ranking Demonstration ===")
    test_queries = [
        "professional resume",
        "birthday invite",
        "business card",
        "wedding invitation",
        "social media post",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = rank_templates(query, model, template_db, k=3)

        for i, (template_id, score) in enumerate(results, 1):
            # Find template details
            template = next(t for t in template_db if t["template_id"] == template_id)
            print(f"  {i}. [{template_id}] {score:.3f} - {template['title']}")

    print("\n🎉 Pipeline execution completed successfully!")
