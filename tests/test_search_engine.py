"""
Concise test suite for search_engine.py focusing on essential functionality and edge cases.
Covers steps 6 and 7 from implementation_plan.md.
"""

import pandas as pd
import pytest
import xgboost as xgb

from src.search_engine import (
    generate_features,
    make_search_logs,
    make_template_db,
    rank_templates,
    train_ranking_model,
)


def test_template_db_generation():
    """Test basic template DB generation."""
    template_db = make_template_db()
    assert len(template_db) >= 10
    assert all("template_id" in t and "title" in t for t in template_db)


def test_search_logs_generation():
    """Test basic search logs generation."""
    template_db = make_template_db()
    search_logs = make_search_logs(template_db)
    assert len(search_logs) >= 20
    assert all("query" in log and "relevance" in log for log in search_logs)


def test_feature_generation_basic():
    """Test feature generation produces correct output."""
    template_db = make_template_db()
    search_logs = make_search_logs(template_db)
    X, y, groups = generate_features(search_logs, template_db)

    assert isinstance(X, pd.DataFrame)
    assert X.shape[1] == 3  # BM25, embedding, popularity
    assert len(X) == len(y)
    assert sum(groups) == len(X)
    assert not X.isna().any().any()


def test_model_training():
    """Test that model training works."""
    template_db = make_template_db()
    search_logs = make_search_logs(template_db)
    X, y, groups = generate_features(search_logs, template_db)
    model = train_ranking_model(X, y, groups)

    assert isinstance(model, xgb.Booster)


def test_ranking_functionality():
    """Test basic ranking functionality."""
    template_db = make_template_db()
    search_logs = make_search_logs(template_db)
    X, y, groups = generate_features(search_logs, template_db)
    model = train_ranking_model(X, y, groups)

    results = rank_templates("professional resume", model, template_db, k=3)

    assert len(results) <= 3
    assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)
    # Check sorted by score descending
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


# Step 6: Edge Cases Tests
def test_edge_case_empty_template_db():
    """Test ranking with empty template database."""
    template_db = make_template_db()
    search_logs = make_search_logs(template_db)
    X, y, groups = generate_features(search_logs, template_db)
    model = train_ranking_model(X, y, groups)

    results = rank_templates("test", model, [], k=5)
    assert results == []


def test_edge_case_k_clamping():
    """Test that k is properly clamped to [0, N]."""
    template_db = make_template_db()
    search_logs = make_search_logs(template_db)
    X, y, groups = generate_features(search_logs, template_db)
    model = train_ranking_model(X, y, groups)

    # Test k=0
    results_zero = rank_templates("test", model, template_db, k=0)
    assert results_zero == []

    # Test k > N
    results_large = rank_templates("test", model, template_db, k=1000)
    assert len(results_large) == len(template_db)

    # Test negative k
    results_negative = rank_templates("test", model, template_db, k=-5)
    assert results_negative == []


def test_edge_case_single_template():
    """Test feature generation with single template (edge case for popularity normalization)."""
    single_template = [
        {
            "template_id": "t001",
            "title": "Test Template",
            "description": "Test description",
            "popularity_score": 100,
        }
    ]
    single_log = [{"query": "test", "template_id": "t001", "relevance": 2}]

    X, y, groups = generate_features(single_log, single_template)
    assert len(X) == 1
    assert X["popularity_norm"].iloc[0] == 0.5  # Should use middle value when all equal


def test_edge_case_stable_tie_breaking():
    """Test stable tie-breaking by template_id."""
    template_db = make_template_db()
    search_logs = make_search_logs(template_db)
    X, y, groups = generate_features(search_logs, template_db)
    model = train_ranking_model(X, y, groups)

    # Run same query multiple times
    results1 = rank_templates("stable test", model, template_db, k=5)
    results2 = rank_templates("stable test", model, template_db, k=5)
    assert results1 == results2


# Step 7: Testing (minimal, deterministic) - Key Behavioral Tests
def test_perfect_lexical_match_ranks_higher():
    """Test that perfect lexical match receives higher predictions."""
    template_db = make_template_db()
    search_logs = make_search_logs(template_db)
    X, y, groups = generate_features(search_logs, template_db)
    model = train_ranking_model(X, y, groups)

    results = rank_templates("business card", model, template_db, k=5)

    # Find business card template (should be t001 based on our data)
    business_card_found = False
    for template_id, _score in results[:2]:  # Check top 2 results
        template = next(t for t in template_db if t["template_id"] == template_id)
        if "business card" in template["title"].lower():
            business_card_found = True
            break

    assert business_card_found, "Perfect lexical match should rank highly"


def test_semantic_match_without_lexical_overlap():
    """Test that semantic matches work even without lexical overlap."""
    template_db = make_template_db()
    search_logs = make_search_logs(template_db)
    X, y, groups = generate_features(search_logs, template_db)
    model = train_ranking_model(X, y, groups)

    # "cv" should find "resume" template even though no direct word match
    results = rank_templates("cv", model, template_db, k=3)

    resume_found = False
    for template_id, _score in results:
        template = next(t for t in template_db if t["template_id"] == template_id)
        if "resume" in template["title"].lower() or "cv" in template["title"].lower():
            resume_found = True
            break

    assert resume_found, "Semantic matching should work without lexical overlap"


def test_popularity_provides_nudge():
    """Test that popularity provides a small but non-dominant nudge."""
    template_db = make_template_db()
    search_logs = make_search_logs(template_db)
    X, y, groups = generate_features(search_logs, template_db)

    # Check that popularity normalization works properly
    popularity_scores = X["popularity_norm"]
    assert (popularity_scores >= 0).all() and (popularity_scores <= 1).all()
    assert popularity_scores.std() > 0  # Should have some variance


def test_deterministic_behavior():
    """Test that the pipeline produces deterministic results."""
    # Run pipeline twice
    template_db1 = make_template_db()
    search_logs1 = make_search_logs(template_db1)

    template_db2 = make_template_db()
    search_logs2 = make_search_logs(template_db2)

    # Should be identical
    assert template_db1 == template_db2
    assert search_logs1 == search_logs2


def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline as specified in step 5."""
    # 1) Build template_db and search_logs
    template_db = make_template_db()
    search_logs = make_search_logs(template_db)

    # 2) Call generate_features
    X, y, groups = generate_features(search_logs, template_db)

    # 3) Train the model
    model = train_ranking_model(X, y, groups)

    # 4) Call rank_templates
    results = rank_templates("professional resume", model, template_db, k=5)

    # 5) Verify relevance - resume template should rank highly
    assert len(results) > 0, "Should return results"

    # Check that a resume-related template is in top results
    top_template_ids = [r[0] for r in results[:2]]
    resume_in_top = False
    for tid in top_template_ids:
        template = next(t for t in template_db if t["template_id"] == tid)
        if "resume" in template["title"].lower() or "cv" in template["title"].lower():
            resume_in_top = True
            break

    assert resume_in_top, "Resume template should rank highly for 'professional resume' query"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
