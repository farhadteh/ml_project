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
        norm_score = (row["popularity_score"] - min_popularity) / (max_popularity - min_popularity)
        popularity_norm.append(norm_score)

    merged_df["popularity_norm"] = popularity_norm

    # Prepare outputs
    X = merged_df[["bm25_score", "embedding_similarity", "popularity_norm"]]
    y = merged_df["relevance"]

    # Calculate group sizes for XGBRanker
    groups = merged_df.groupby("query").size().tolist()

    return X, y, groups


def run_tests() -> None:
    """Basic tests for the feature generation pipeline."""
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
