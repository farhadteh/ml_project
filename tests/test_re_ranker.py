"""
Test suite for the gender diversity re-ranking system.

This module contains comprehensive tests for all re-ranking strategies,
validation checks, and edge cases.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from re_ranker import (
    evaluate_diversity_score,
    generate_mock_data,
    generate_mock_data_extreme_female,
    generate_mock_data_extreme_male,
    generate_mock_data_female_bias,
    rerank_boosted_demotion,
    rerank_interleave,
    rerank_proportional,
)


def test_mock_data_generation():
    """Test mock data generation functionality."""
    print("Testing mock data generation...")

    data = generate_mock_data(seed=42)

    # Basic structure tests
    assert len(data) == 100, f"Expected 100 items, got {len(data)}"
    assert all(
        "item_id" in item and "relevance_score" in item and "gender" in item for item in data
    ), "All items should have required keys"

    # Test reproducibility
    data2 = generate_mock_data(seed=42)
    assert data == data2, "Same seed should produce identical results"

    # Test different seed produces different results
    data3 = generate_mock_data(seed=123)
    assert data != data3, "Different seed should produce different results"

    # Test relevance score ordering
    top_10 = sorted(data, key=lambda x: x["relevance_score"], reverse=True)[:10]
    for i in range(9):
        assert (
            top_10[i]["relevance_score"] >= top_10[i + 1]["relevance_score"]
        ), "Relevance scores should be in descending order"

    print("✓ Mock data generation tests passed")


def test_diversity_evaluation():
    """Test diversity score evaluation."""
    print("Testing diversity evaluation...")

    # Test with known data
    test_data = [
        {"gender": "Male", "item_id": "1"},
        {"gender": "Female", "item_id": "2"},
        {"gender": "Male", "item_id": "3"},
        {"gender": "Female", "item_id": "4"},
        {"gender": "Female", "item_id": "5"},
    ]

    score = evaluate_diversity_score(test_data)
    assert score == 3, f"Expected diversity score 3, got {score}"

    # Test with all male
    all_male = [{"gender": "Male", "item_id": str(i)} for i in range(10)]
    score = evaluate_diversity_score(all_male)
    assert score == 0, f"Expected diversity score 0 for all male, got {score}"

    # Test with more than 10 items (should only consider first 10)
    extended_data = test_data + [{"gender": "Female", "item_id": str(i)} for i in range(10)]
    score = evaluate_diversity_score(extended_data)
    # The first 10 items include our original 5 (with 3 females) + 5 more females = 8 total females
    assert score == 8, f"Should only consider first 10 items, got {score}"

    print("✓ Diversity evaluation tests passed")


def test_baseline_analysis():
    """Test baseline analysis produces expected imbalance."""
    print("Testing baseline analysis...")

    data = generate_mock_data(seed=42)
    baseline_top_10 = sorted(data, key=lambda x: x["relevance_score"], reverse=True)[:10]
    baseline_diversity = evaluate_diversity_score(baseline_top_10)

    # Should have low diversity (biased toward males)
    assert baseline_diversity <= 3, f"Baseline should have low diversity, got {baseline_diversity}"

    # Count gender distribution
    gender_counts = {}
    for item in baseline_top_10:
        gender_counts[item["gender"]] = gender_counts.get(item["gender"], 0) + 1

    male_count = gender_counts.get("Male", 0)
    female_count = gender_counts.get("Female", 0)

    assert male_count > female_count, "Baseline should be biased toward males"
    print(f"Male-biased baseline: {male_count}M:{female_count}F (diversity: {baseline_diversity})")

    print("✓ Baseline analysis tests passed")


def test_female_biased_baseline():
    """Test female-biased baseline analysis produces expected imbalance."""
    print("Testing female-biased baseline analysis...")

    data = generate_mock_data_female_bias(seed=42)
    baseline_top_10 = sorted(data, key=lambda x: x["relevance_score"], reverse=True)[:10]
    baseline_diversity = evaluate_diversity_score(baseline_top_10)

    # Should have high diversity (biased toward females)
    assert (
        baseline_diversity >= 7
    ), f"Female-biased baseline should have high diversity, got {baseline_diversity}"

    # Count gender distribution
    gender_counts = {}
    for item in baseline_top_10:
        gender_counts[item["gender"]] = gender_counts.get(item["gender"], 0) + 1

    male_count = gender_counts.get("Male", 0)
    female_count = gender_counts.get("Female", 0)

    assert female_count > male_count, "Female-biased baseline should be biased toward females"
    print(
        f"Female-biased baseline: {male_count}M:{female_count}F (diversity: {baseline_diversity})"
    )

    # Test that re-ranking works in reverse (increasing male representation)
    interleaved = rerank_interleave(data)
    male_count_after = sum(1 for item in interleaved if item["gender"] == "Male")
    print(f"After interleaving: {male_count_after} males (vs {male_count} baseline)")

    print("✓ Female-biased baseline analysis tests passed")


def test_extreme_scenarios():
    """Test extreme bias scenarios where all top-10 are single gender."""
    print("Testing extreme bias scenarios...")

    # Test extreme male bias (all top-10 male)
    extreme_male_data = generate_mock_data_extreme_male(seed=42)
    extreme_male_top_10 = sorted(
        extreme_male_data, key=lambda x: x["relevance_score"], reverse=True
    )[:10]
    extreme_male_diversity = evaluate_diversity_score(extreme_male_top_10)

    assert (
        extreme_male_diversity == 0
    ), f"Extreme male bias should have 0 diversity, got {extreme_male_diversity}"

    # Test that all top-10 are male
    all_male = all(item["gender"] == "Male" for item in extreme_male_top_10)
    assert all_male, "All top-10 items should be male in extreme male scenario"

    # Test re-ranking works on extreme male bias
    male_interleaved = rerank_interleave(extreme_male_data)
    male_interleaved_diversity = evaluate_diversity_score(male_interleaved)
    assert (
        male_interleaved_diversity > 0
    ), "Interleaving should introduce female items from extreme male bias"

    male_proportional = rerank_proportional(extreme_male_data)
    male_proportional_diversity = evaluate_diversity_score(male_proportional)
    assert (
        male_proportional_diversity == 5
    ), "Proportional should achieve 5 female items even from extreme male bias"

    print(f"Extreme male: 0 → {male_interleaved_diversity} female (interleaving)")
    print(f"Extreme male: 0 → {male_proportional_diversity} female (proportional)")

    # Test extreme female bias (all top-10 female)
    extreme_female_data = generate_mock_data_extreme_female(seed=42)
    extreme_female_top_10 = sorted(
        extreme_female_data, key=lambda x: x["relevance_score"], reverse=True
    )[:10]
    extreme_female_diversity = evaluate_diversity_score(extreme_female_top_10)

    assert (
        extreme_female_diversity == 10
    ), f"Extreme female bias should have 10 diversity, got {extreme_female_diversity}"

    # Test that all top-10 are female
    all_female = all(item["gender"] == "Female" for item in extreme_female_top_10)
    assert all_female, "All top-10 items should be female in extreme female scenario"

    # Test re-ranking works on extreme female bias (count males)
    def count_males(results):
        return sum(1 for item in results if item["gender"] == "Male")

    female_baseline_males = count_males(extreme_female_top_10)
    assert female_baseline_males == 0, "Extreme female scenario should have 0 males in baseline"

    female_interleaved = rerank_interleave(extreme_female_data)
    female_interleaved_males = count_males(female_interleaved)
    assert (
        female_interleaved_males > 0
    ), "Interleaving should introduce male items from extreme female bias"

    female_proportional = rerank_proportional(extreme_female_data)
    female_proportional_males = count_males(female_proportional)
    assert (
        female_proportional_males == 5
    ), "Proportional should achieve 5 male items even from extreme female bias"

    print(f"Extreme female: 0 → {female_interleaved_males} male (interleaving)")
    print(f"Extreme female: 0 → {female_proportional_males} male (proportional)")

    print("✓ Extreme bias scenario tests passed")


def test_simple_interleaving():
    """Test simple interleaving strategy."""
    print("Testing simple interleaving strategy...")

    data = generate_mock_data(seed=42)
    baseline_diversity = evaluate_diversity_score(
        sorted(data, key=lambda x: x["relevance_score"], reverse=True)[:10]
    )

    interleaved = rerank_interleave(data)
    interleaved_diversity = evaluate_diversity_score(interleaved)

    # Basic validation
    assert len(interleaved) == 10, f"Should return 10 items, got {len(interleaved)}"
    assert all(
        "item_id" in item and "relevance_score" in item and "gender" in item for item in interleaved
    ), "All items should have required keys"

    # Should improve diversity
    assert interleaved_diversity > baseline_diversity, (
        f"Should improve diversity: baseline={baseline_diversity}, "
        f"interleaved={interleaved_diversity}"
    )

    # Should achieve target
    assert interleaved_diversity >= 3, f"Should achieve target ≥3, got {interleaved_diversity}"

    # Check alternating pattern in early results
    early_genders = [item["gender"] for item in interleaved[:6]]
    print(f"Early interleaving pattern: {early_genders}")

    print(f"✓ Simple interleaving tests passed (diversity: {interleaved_diversity})")


def test_boosted_demotion():
    """Test boosted demotion strategy."""
    print("Testing boosted demotion strategy...")

    data = generate_mock_data(seed=42)
    baseline_diversity = evaluate_diversity_score(
        sorted(data, key=lambda x: x["relevance_score"], reverse=True)[:10]
    )

    # Test with different alpha values
    alpha_values = [0.0, 0.1, 0.2, 0.5]

    for alpha in alpha_values:
        boosted = rerank_boosted_demotion(data, alpha=alpha)
        boosted_diversity = evaluate_diversity_score(boosted)

        assert len(boosted) == 10, f"Should return 10 items for alpha={alpha}"
        assert boosted_diversity >= baseline_diversity, (
            f"Should maintain/improve diversity for alpha={alpha}: "
            f"baseline={baseline_diversity}, boosted={boosted_diversity}"
        )

        print(f"Alpha {alpha}: diversity={boosted_diversity}")

    # Test parameter validation
    try:
        rerank_boosted_demotion(data, alpha=-0.1)
        raise AssertionError("Should raise error for negative alpha")
    except ValueError:
        pass

    try:
        rerank_boosted_demotion(data, alpha=1.5)
        raise AssertionError("Should raise error for alpha > 1.0")
    except ValueError:
        pass

    print("✓ Boosted demotion tests passed")


def test_proportional_ranking():
    """Test proportional re-ranking strategy."""
    print("Testing proportional re-ranking strategy...")

    data = generate_mock_data(seed=42)
    baseline_diversity = evaluate_diversity_score(
        sorted(data, key=lambda x: x["relevance_score"], reverse=True)[:10]
    )

    proportional = rerank_proportional(data)
    proportional_diversity = evaluate_diversity_score(proportional)

    # Basic validation
    assert len(proportional) == 10, f"Should return 10 items, got {len(proportional)}"
    assert proportional_diversity >= baseline_diversity, (
        f"Should maintain/improve diversity: baseline={baseline_diversity}, "
        f"proportional={proportional_diversity}"
    )

    # Should achieve target
    assert proportional_diversity >= 3, f"Should achieve target ≥3, got {proportional_diversity}"

    # Check gender distribution is balanced
    gender_counts = {}
    for item in proportional:
        gender_counts[item["gender"]] = gender_counts.get(item["gender"], 0) + 1

    male_count = gender_counts.get("Male", 0)
    female_count = gender_counts.get("Female", 0)

    print(f"Proportional distribution: {male_count}M:{female_count}F")

    # Should be approximately balanced (allowing for data constraints)
    assert (
        abs(male_count - female_count) <= 2
    ), f"Should have balanced distribution, got {male_count}M:{female_count}F"

    print(f"✓ Proportional ranking tests passed (diversity: {proportional_diversity})")


def test_strategy_comparison():
    """Test and compare all strategies."""
    print("Testing strategy comparison...")

    data = generate_mock_data(seed=42)

    # Get baseline
    baseline_top_10 = sorted(data, key=lambda x: x["relevance_score"], reverse=True)[:10]
    baseline_diversity = evaluate_diversity_score(baseline_top_10)
    baseline_avg_relevance = sum(item["relevance_score"] for item in baseline_top_10) / 10

    # Apply all strategies
    strategies = {
        "Simple Interleaving": rerank_interleave(data),
        "Boosted Demotion": rerank_boosted_demotion(data, alpha=0.2),
        "Proportional": rerank_proportional(data),
    }

    print("\nStrategy Comparison:")
    print(f"Baseline: diversity={baseline_diversity}, avg_relevance={baseline_avg_relevance:.3f}")

    for name, results in strategies.items():
        diversity = evaluate_diversity_score(results)
        avg_relevance = sum(item["relevance_score"] for item in results) / 10
        improvement = diversity - baseline_diversity

        print(f"{name}: diversity={diversity} (+{improvement}), avg_relevance={avg_relevance:.3f}")

        # All strategies should improve or maintain diversity
        assert diversity >= baseline_diversity, f"{name} should maintain/improve diversity"

        # All strategies should achieve target
        assert diversity >= 3, f"{name} should achieve target ≥3"

    print("✓ Strategy comparison tests passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")

    # Test with minimal data
    minimal_data = [
        {"item_id": "1", "relevance_score": 1.0, "gender": "Male"},
        {"item_id": "2", "relevance_score": 0.9, "gender": "Female"},
    ]

    for strategy_func in [rerank_interleave, rerank_proportional]:
        result = strategy_func(minimal_data)
        assert len(result) <= 10, "Should handle minimal data gracefully"
        assert len(result) <= len(minimal_data), "Should not create extra items"

    # Test boosted demotion with minimal data
    result = rerank_boosted_demotion(minimal_data, alpha=0.1)
    assert len(result) <= 10, "Should handle minimal data gracefully"

    # Test with all same gender
    all_male_data = [
        {"item_id": str(i), "relevance_score": 1.0 - i * 0.1, "gender": "Male"} for i in range(20)
    ]

    for strategy_func in [rerank_interleave, rerank_proportional]:
        result = strategy_func(all_male_data)
        assert len(result) == 10, "Should return 10 items even with single gender"
        diversity = evaluate_diversity_score(result)
        assert diversity == 0, "Should have 0 diversity with single gender"

    print("✓ Edge case tests passed")


def run_all_tests():
    """Run all test functions."""
    print("=" * 60)
    print("RUNNING COMPREHENSIVE RE-RANKER TESTS")
    print("=" * 60)

    test_functions = [
        test_mock_data_generation,
        test_diversity_evaluation,
        test_baseline_analysis,
        test_female_biased_baseline,
        test_extreme_scenarios,
        test_simple_interleaving,
        test_boosted_demotion,
        test_proportional_ranking,
        test_strategy_comparison,
        test_edge_cases,
    ]

    passed = 0
    total = len(test_functions)

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            continue

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Re-ranking system is working correctly.")
    else:
        print(f"❌ {total - passed} tests failed. Please review the errors above.")

    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
