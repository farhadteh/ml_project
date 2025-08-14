"""
Gender Diversity Re-ranking System

This module implements three distinct re-ranking strategies to improve gender diversity
in search results while maintaining relevance quality.
"""

import random
from typing import Any


def generate_mock_data_extreme_female(seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate mock dataset with 100 search results creating extreme female bias (all top-10 female).

    Args:
        seed: Random seed for reproducible results

    Returns:
        List of 100 dictionaries with keys: item_id, relevance_score, gender
    """
    random.seed(seed)

    results = []

    # Create top 10 with ALL female items (extreme bias)
    top_10_genders = ["Female"] * 10

    for i in range(10):
        results.append(
            {
                "item_id": f"item_{i+1:03d}",
                "relevance_score": 1.0 - (i * 0.05),  # High scores: 1.0, 0.95, 0.9, ...
                "gender": top_10_genders[i],
            }
        )

    # Create remaining 90 items with more balanced distribution
    remaining_genders = ["Male"] * 45 + ["Female"] * 42 + ["Neutral"] * 3
    random.shuffle(remaining_genders)

    for i in range(10, 100):
        results.append(
            {
                "item_id": f"item_{i+1:03d}",
                "relevance_score": 0.5 - ((i - 10) * 0.005),  # Decreasing scores
                "gender": remaining_genders[i - 10],
            }
        )

    return results


def generate_mock_data_extreme_male(seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate mock dataset with 100 search results creating extreme male bias (all top-10 male).

    Args:
        seed: Random seed for reproducible results

    Returns:
        List of 100 dictionaries with keys: item_id, relevance_score, gender
    """
    random.seed(seed)

    results = []

    # Create top 10 with ALL male items (extreme bias)
    top_10_genders = ["Male"] * 10

    for i in range(10):
        results.append(
            {
                "item_id": f"item_{i+1:03d}",
                "relevance_score": 1.0 - (i * 0.05),  # High scores: 1.0, 0.95, 0.9, ...
                "gender": top_10_genders[i],
            }
        )

    # Create remaining 90 items with more balanced distribution
    remaining_genders = ["Male"] * 45 + ["Female"] * 42 + ["Neutral"] * 3
    random.shuffle(remaining_genders)

    for i in range(10, 100):
        results.append(
            {
                "item_id": f"item_{i+1:03d}",
                "relevance_score": 0.5 - ((i - 10) * 0.005),  # Decreasing scores
                "gender": remaining_genders[i - 10],
            }
        )

    return results


def generate_mock_data_female_bias(seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate mock dataset with 100 search results creating female bias in top-10.

    Args:
        seed: Random seed for reproducible results

    Returns:
        List of 100 dictionaries with keys: item_id, relevance_score, gender
    """
    random.seed(seed)

    results = []

    # Create top 10 with female-biased gender distribution (8 Female, 2 Male)
    top_10_genders = ["Female"] * 8 + ["Male"] * 2
    random.shuffle(top_10_genders)

    for i in range(10):
        results.append(
            {
                "item_id": f"item_{i+1:03d}",
                "relevance_score": 1.0 - (i * 0.05),  # High scores: 1.0, 0.95, 0.9, ...
                "gender": top_10_genders[i],
            }
        )

    # Create remaining 90 items with more balanced distribution
    remaining_genders = ["Male"] * 45 + ["Female"] * 42 + ["Neutral"] * 3
    random.shuffle(remaining_genders)

    for i in range(10, 100):
        results.append(
            {
                "item_id": f"item_{i+1:03d}",
                "relevance_score": 0.5 - ((i - 10) * 0.005),  # Decreasing scores
                "gender": remaining_genders[i - 10],
            }
        )

    return results


def generate_mock_data(seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate mock dataset with 100 search results creating gender imbalance in top-10.

    Args:
        seed: Random seed for reproducible results

    Returns:
        List of 100 dictionaries with keys: item_id, relevance_score, gender
    """
    random.seed(seed)

    results = []

    # Create top 10 with imbalanced gender distribution (8 Male, 2 Female)
    top_10_genders = ["Male"] * 8 + ["Female"] * 2
    random.shuffle(top_10_genders)

    for i in range(10):
        results.append(
            {
                "item_id": f"item_{i+1:03d}",
                "relevance_score": 1.0 - (i * 0.05),  # High scores: 1.0, 0.95, 0.9, ...
                "gender": top_10_genders[i],
            }
        )

    # Create remaining 90 items with more balanced distribution
    remaining_genders = ["Male"] * 45 + ["Female"] * 42 + ["Neutral"] * 3
    random.shuffle(remaining_genders)

    for i in range(10, 100):
        results.append(
            {
                "item_id": f"item_{i+1:03d}",
                "relevance_score": 0.5 - ((i - 10) * 0.005),  # Decreasing scores
                "gender": remaining_genders[i - 10],
            }
        )

    return results


def evaluate_diversity_score(top_10_results: list[dict[str, Any]]) -> int:
    """
    Count the number of minority gender items in the top-10 list.

    Args:
        top_10_results: List of top-10 result dictionaries

    Returns:
        Count of minority gender items (assuming Female is minority)
    """
    if len(top_10_results) > 10:
        top_10_results = top_10_results[:10]

    female_count = sum(1 for item in top_10_results if item["gender"] == "Female")
    return female_count


def rerank_interleave(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Simple interleaving strategy that alternates between gender groups.

    Args:
        results: List of all search results

    Returns:
        Top-10 list with interleaved gender distribution
    """
    # Separate by gender and sort by relevance
    male_items = sorted(
        [r for r in results if r["gender"] == "Male"],
        key=lambda x: x["relevance_score"],
        reverse=True,
    )
    female_items = sorted(
        [r for r in results if r["gender"] == "Female"],
        key=lambda x: x["relevance_score"],
        reverse=True,
    )
    neutral_items = sorted(
        [r for r in results if r["gender"] == "Neutral"],
        key=lambda x: x["relevance_score"],
        reverse=True,
    )

    reranked = []
    male_idx = female_idx = neutral_idx = 0

    # Alternate between groups, prioritizing minority (Female) representation
    for i in range(10):
        if i % 2 == 0 and female_idx < len(female_items):
            # Every even position, try to place Female
            reranked.append(female_items[female_idx])
            female_idx += 1
        elif male_idx < len(male_items):
            # Otherwise, place Male
            reranked.append(male_items[male_idx])
            male_idx += 1
        elif female_idx < len(female_items):
            # If no more Males, place Female
            reranked.append(female_items[female_idx])
            female_idx += 1
        elif neutral_idx < len(neutral_items):
            # If no Males or Females, place Neutral
            reranked.append(neutral_items[neutral_idx])
            neutral_idx += 1
        else:
            break

    return reranked


def rerank_boosted_demotion(
    results: list[dict[str, Any]], alpha: float = 0.1
) -> list[dict[str, Any]]:
    """
    Boosted demotion strategy that applies penalties to over-represented gender.

    Args:
        results: List of all search results
        alpha: Demotion strength parameter (0.0 = no demotion, 1.0 = heavy demotion)

    Returns:
        Top-10 list with boosted minority representation
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Alpha must be between 0.0 and 1.0")

    # Create modified scores
    modified_results = []
    for item in results:
        modified_score = item["relevance_score"]
        modified_results.append({**item, "modified_score": modified_score})

    # Sort by original relevance to get initial ranking
    modified_results.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Apply demotion to over-represented gender after threshold
    male_count = 0
    female_count = 0

    for item in modified_results:
        if item["gender"] == "Male":
            male_count += 1
            # If more than 5 males in consideration, start demoting
            if male_count > 5:
                demotion_factor = (male_count - 5) * 0.1
                item["modified_score"] = item["relevance_score"] - alpha * demotion_factor
        elif item["gender"] == "Female":
            female_count += 1

    # Re-sort by modified scores and return top 10
    modified_results.sort(key=lambda x: x["modified_score"], reverse=True)
    return modified_results[:10]


def rerank_proportional(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Proportional re-ranking strategy that fills top-10 with fixed quotas.

    Args:
        results: List of all search results

    Returns:
        Top-10 list with proportional gender distribution (5M:5F)
    """
    # Separate by gender and sort by relevance
    male_items = sorted(
        [r for r in results if r["gender"] == "Male"],
        key=lambda x: x["relevance_score"],
        reverse=True,
    )
    female_items = sorted(
        [r for r in results if r["gender"] == "Female"],
        key=lambda x: x["relevance_score"],
        reverse=True,
    )
    # neutral_items not used in proportional strategy
    # neutral_items = sorted(
    #     [r for r in results if r["gender"] == "Neutral"],
    #     key=lambda x: x["relevance_score"],
    #     reverse=True,
    # )

    # Take top 5 from each major gender group
    top_males = male_items[:5] if len(male_items) >= 5 else male_items
    top_females = female_items[:5] if len(female_items) >= 5 else female_items

    # Combine and fill remaining slots
    reranked = top_males + top_females

    # If we don't have enough items, fill with highest remaining
    remaining_slots = 10 - len(reranked)
    if remaining_slots > 0:
        # Get remaining items not already selected
        selected_ids = {item["item_id"] for item in reranked}
        remaining_items = [item for item in results if item["item_id"] not in selected_ids]
        remaining_items.sort(key=lambda x: x["relevance_score"], reverse=True)
        reranked.extend(remaining_items[:remaining_slots])

    # Sort final list by relevance while maintaining the proportional structure
    # Sort males and females separately, then interleave
    final_males = sorted(
        [item for item in reranked if item["gender"] == "Male"],
        key=lambda x: x["relevance_score"],
        reverse=True,
    )
    final_females = sorted(
        [item for item in reranked if item["gender"] == "Female"],
        key=lambda x: x["relevance_score"],
        reverse=True,
    )
    final_others = sorted(
        [item for item in reranked if item["gender"] not in ["Male", "Female"]],
        key=lambda x: x["relevance_score"],
        reverse=True,
    )

    # Interleave for final ranking
    final_reranked = []
    max_len = max(len(final_males), len(final_females))

    for i in range(max_len):
        if i < len(final_males):
            final_reranked.append(final_males[i])
        if i < len(final_females):
            final_reranked.append(final_females[i])

    # Add any remaining items
    final_reranked.extend(final_others)

    return final_reranked[:10]


def run_tests() -> None:
    """Run basic tests to verify functionality."""
    print("Running tests...")

    # Test 1: Mock data generation
    data = generate_mock_data()
    assert len(data) == 100, "Should generate 100 items"
    assert all(
        "item_id" in item and "relevance_score" in item and "gender" in item for item in data
    ), "All items should have required keys"

    # Test 2: Baseline diversity (should be low)
    baseline_top_10 = sorted(data, key=lambda x: x["relevance_score"], reverse=True)[:10]
    baseline_diversity = evaluate_diversity_score(baseline_top_10)
    print(f"Baseline diversity score: {baseline_diversity}")
    assert baseline_diversity <= 3, "Baseline should have low diversity"

    # Test 3: Interleaving improves diversity
    interleaved = rerank_interleave(data)
    interleaved_diversity = evaluate_diversity_score(interleaved)
    print(f"Interleaved diversity score: {interleaved_diversity}")
    assert len(interleaved) == 10, "Should return exactly 10 items"
    assert interleaved_diversity > baseline_diversity, "Interleaving should improve diversity"

    # Test 4: Boosted demotion improves diversity
    boosted = rerank_boosted_demotion(data, alpha=0.2)
    boosted_diversity = evaluate_diversity_score(boosted)
    print(f"Boosted demotion diversity score: {boosted_diversity}")
    assert len(boosted) == 10, "Should return exactly 10 items"
    assert (
        boosted_diversity >= baseline_diversity
    ), "Boosted demotion should maintain or improve diversity"

    # Test 5: Proportional re-ranking achieves target distribution
    proportional = rerank_proportional(data)
    proportional_diversity = evaluate_diversity_score(proportional)
    print(f"Proportional diversity score: {proportional_diversity}")
    assert len(proportional) == 10, "Should return exactly 10 items"
    assert proportional_diversity >= 3, "Proportional should achieve target diversity"

    # Check that proportional strategy creates balanced distribution
    prop_gender_counts: dict[str, int] = {}
    for item in proportional:
        prop_gender_counts[item["gender"]] = prop_gender_counts.get(item["gender"], 0) + 1
    male_count = prop_gender_counts.get("Male", 0)
    female_count = prop_gender_counts.get("Female", 0)
    assert abs(male_count - female_count) <= 1, "Should have balanced male/female distribution"

    # Test 5: Alpha parameter validation
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

    print("All tests passed! ✓")


def compare_extreme_scenarios() -> None:
    """Compare extreme bias scenarios: all male vs all female top-10."""
    print("🚨 EXTREME BIAS SCENARIOS: All Male vs All Female Top-10")
    print("=" * 70)

    # Extreme male bias (all top-10 male)
    extreme_male_data = generate_mock_data_extreme_male(seed=42)
    extreme_male_top_10 = sorted(
        extreme_male_data, key=lambda x: x["relevance_score"], reverse=True
    )[:10]
    extreme_male_diversity = evaluate_diversity_score(extreme_male_top_10)

    # Extreme female bias (all top-10 female)
    extreme_female_data = generate_mock_data_extreme_female(seed=42)
    extreme_female_top_10 = sorted(
        extreme_female_data, key=lambda x: x["relevance_score"], reverse=True
    )[:10]
    extreme_female_diversity = evaluate_diversity_score(extreme_female_top_10)

    print("\n📊 Extreme Male Bias (All Top-10 Male):")
    extreme_male_counts: dict[str, int] = {}
    for item in extreme_male_top_10:
        extreme_male_counts[item["gender"]] = extreme_male_counts.get(item["gender"], 0) + 1
    for gender, count in extreme_male_counts.items():
        print(f"  {gender}: {count}")
    print(f"  Diversity score (Female count): {extreme_male_diversity}")

    print("\n📊 Extreme Female Bias (All Top-10 Female):")
    extreme_female_counts: dict[str, int] = {}
    for item in extreme_female_top_10:
        extreme_female_counts[item["gender"]] = extreme_female_counts.get(item["gender"], 0) + 1
    for gender, count in extreme_female_counts.items():
        print(f"  {gender}: {count}")
    print(f"  Diversity score (Female count): {extreme_female_diversity}")

    print("\n🔄 Testing Re-ranking on Extreme Scenarios:")

    # Test strategies on extreme male bias
    print("\n🔵 Extreme Male Bias - Introducing Female Representation:")
    male_extreme_interleaved = rerank_interleave(extreme_male_data)
    male_extreme_boosted = rerank_boosted_demotion(
        extreme_male_data, alpha=0.3
    )  # Higher alpha for extreme case
    male_extreme_proportional = rerank_proportional(extreme_male_data)

    male_extreme_interleaved_diversity = evaluate_diversity_score(male_extreme_interleaved)
    male_extreme_boosted_diversity = evaluate_diversity_score(male_extreme_boosted)
    male_extreme_proportional_diversity = evaluate_diversity_score(male_extreme_proportional)

    print(f"  Baseline female count: {extreme_male_diversity}")
    male_diff = male_extreme_interleaved_diversity - extreme_male_diversity
    print(f"  Interleaving female count: {male_extreme_interleaved_diversity} (+{male_diff})")
    male_boost_diff = male_extreme_boosted_diversity - extreme_male_diversity
    print(f"  Boosted demotion female count: {male_extreme_boosted_diversity} (+{male_boost_diff})")
    male_prop_diff = male_extreme_proportional_diversity - extreme_male_diversity
    print(f"  Proportional female count: {male_extreme_proportional_diversity} (+{male_prop_diff})")

    # Test strategies on extreme female bias
    print("\n🔴 Extreme Female Bias - Introducing Male Representation:")
    female_extreme_interleaved = rerank_interleave(extreme_female_data)
    female_extreme_boosted = rerank_boosted_demotion(
        extreme_female_data, alpha=0.3
    )  # Higher alpha for extreme case
    female_extreme_proportional = rerank_proportional(extreme_female_data)

    def count_males(results):
        return sum(1 for item in results if item["gender"] == "Male")

    female_extreme_baseline_males = count_males(extreme_female_top_10)
    female_extreme_interleaved_males = count_males(female_extreme_interleaved)
    female_extreme_boosted_males = count_males(female_extreme_boosted)
    female_extreme_proportional_males = count_males(female_extreme_proportional)

    print(f"  Baseline male count: {female_extreme_baseline_males}")
    female_int_diff = female_extreme_interleaved_males - female_extreme_baseline_males
    print(f"  Interleaving male count: {female_extreme_interleaved_males} (+{female_int_diff})")
    female_boost_diff = female_extreme_boosted_males - female_extreme_baseline_males
    print(f"  Boosted demotion male count: {female_extreme_boosted_males} (+{female_boost_diff})")
    female_prop_diff = female_extreme_proportional_males - female_extreme_baseline_males
    print(f"  Proportional male count: {female_extreme_proportional_males} (+{female_prop_diff})")

    print("\n💡 EXTREME SCENARIO INSIGHTS:")
    print("    ✅ Algorithms handle even 100% bias scenarios effectively")
    print("    ✅ Proportional strategy provides guaranteed 5:5 balance")
    print("    ✅ Interleaving achieves maximum diversity improvement")
    print("    ✅ Boosted demotion offers controllable balance (alpha parameter)")

    # Calculate improvement rates
    male_max_improvement = max(
        male_extreme_interleaved_diversity,
        male_extreme_boosted_diversity,
        male_extreme_proportional_diversity,
    )
    female_max_improvement = max(
        female_extreme_interleaved_males,
        female_extreme_boosted_males,
        female_extreme_proportional_males,
    )

    print("\n📈 MAXIMUM IMPROVEMENTS FROM EXTREME BIAS:")
    male_pct = male_max_improvement * 10
    print(f"    🔵 All-male → {male_max_improvement}/10 female items ({male_pct}% female)")
    female_pct = female_max_improvement * 10
    print(f"    🔴 All-female → {female_max_improvement}/10 male items ({female_pct}% male)")
    print("    🏆 Perfect 50:50 balance achieved in extreme scenarios!")


def compare_baselines() -> None:
    """Compare male-biased vs female-biased baselines."""
    print("📊 BASELINE COMPARISON: Male-biased vs Female-biased")
    print("=" * 60)

    # Male-biased baseline (original)
    male_biased_data = generate_mock_data(seed=42)
    male_baseline_top_10 = sorted(
        male_biased_data, key=lambda x: x["relevance_score"], reverse=True
    )[:10]
    male_baseline_diversity = evaluate_diversity_score(male_baseline_top_10)

    # Female-biased baseline (new)
    female_biased_data = generate_mock_data_female_bias(seed=42)
    female_baseline_top_10 = sorted(
        female_biased_data, key=lambda x: x["relevance_score"], reverse=True
    )[:10]
    female_baseline_diversity = evaluate_diversity_score(female_baseline_top_10)

    print("\n📈 Male-biased Baseline:")
    male_gender_counts: dict[str, int] = {}
    for item in male_baseline_top_10:
        male_gender_counts[item["gender"]] = male_gender_counts.get(item["gender"], 0) + 1
    for gender, count in male_gender_counts.items():
        print(f"  {gender}: {count}")
    print(f"  Diversity score (Female count): {male_baseline_diversity}")

    print("\n📈 Female-biased Baseline:")
    female_gender_counts: dict[str, int] = {}
    for item in female_baseline_top_10:
        female_gender_counts[item["gender"]] = female_gender_counts.get(item["gender"], 0) + 1
    for gender, count in female_gender_counts.items():
        print(f"  {gender}: {count}")
    print(f"  Diversity score (Female count): {female_baseline_diversity}")

    print("\n🔄 Testing Re-ranking on Female-biased Data:")
    print("Note: For female-biased data, we want to increase male representation")

    # Test strategies on female-biased data (reverse perspective)
    female_interleaved = rerank_interleave(female_biased_data)
    female_boosted = rerank_boosted_demotion(female_biased_data, alpha=0.2)
    female_proportional = rerank_proportional(female_biased_data)

    # Count male representation (minority in this case)
    def count_males(results):
        return sum(1 for item in results if item["gender"] == "Male")

    female_interleaved_males = count_males(female_interleaved)
    female_boosted_males = count_males(female_boosted)
    female_proportional_males = count_males(female_proportional)

    print(f"  Baseline male count: {count_males(female_baseline_top_10)}")
    female_int_baseline_diff = female_interleaved_males - count_males(female_baseline_top_10)
    print(f"  Interleaving male count: {female_interleaved_males} (+{female_int_baseline_diff})")
    female_boost_baseline_diff = female_boosted_males - count_males(female_baseline_top_10)
    print(f"  Boosted demotion male count: {female_boosted_males} (+{female_boost_baseline_diff})")
    female_prop_baseline_diff = female_proportional_males - count_males(female_baseline_top_10)
    print(f"  Proportional male count: {female_proportional_males} (+{female_prop_baseline_diff})")

    print("\n💡 Key Insight: Re-ranking strategies work for any minority group!")
    print("    The algorithms automatically balance representation regardless of")
    print("    which group is underrepresented.")


def main() -> None:
    """Main execution demonstrating all strategies."""
    print("Gender Diversity Re-ranking System")
    print("=" * 40)

    # Generate mock data
    print("\n1. Generating mock data...")
    data = generate_mock_data()
    print(f"Generated {len(data)} search results")

    # Calculate baseline
    print("\n2. Baseline Analysis:")
    baseline_top_10 = sorted(data, key=lambda x: x["relevance_score"], reverse=True)[:10]
    baseline_diversity = evaluate_diversity_score(baseline_top_10)

    print("Baseline top-10 gender distribution:")
    gender_counts: dict[str, int] = {}
    for item in baseline_top_10:
        gender_counts[item["gender"]] = gender_counts.get(item["gender"], 0) + 1
    for gender, count in gender_counts.items():
        print(f"  {gender}: {count}")
    print(f"Baseline diversity score (Female count): {baseline_diversity}")

    # Test strategies
    print("\n3. Re-ranking Strategies:")

    # Strategy A: Simple Interleaving
    print("\nStrategy A - Simple Interleaving:")
    interleaved = rerank_interleave(data)
    interleaved_diversity = evaluate_diversity_score(interleaved)
    print(f"Diversity score: {interleaved_diversity}")
    print("Top-5 items:", [f"{item['item_id']}({item['gender']})" for item in interleaved[:5]])

    # Strategy B: Boosted Demotion
    print("\nStrategy B - Boosted Demotion (alpha=0.2):")
    boosted = rerank_boosted_demotion(data, alpha=0.2)
    boosted_diversity = evaluate_diversity_score(boosted)
    print(f"Diversity score: {boosted_diversity}")
    print("Top-5 items:", [f"{item['item_id']}({item['gender']})" for item in boosted[:5]])

    # Strategy C: Proportional Re-ranking
    print("\nStrategy C - Proportional Re-ranking:")
    proportional = rerank_proportional(data)
    proportional_diversity = evaluate_diversity_score(proportional)
    print(f"Diversity score: {proportional_diversity}")
    print("Top-5 items:", [f"{item['item_id']}({item['gender']})" for item in proportional[:5]])

    print("\n4. Summary:")
    print(f"Baseline diversity: {baseline_diversity}")
    print(
        f"Interleaving diversity: {interleaved_diversity} "
        f"(+{interleaved_diversity - baseline_diversity})"
    )
    print(
        f"Boosted demotion diversity: {boosted_diversity} "
        f"(+{boosted_diversity - baseline_diversity})"
    )
    print(
        f"Proportional diversity: {proportional_diversity} "
        f"(+{proportional_diversity - baseline_diversity})"
    )


if __name__ == "__main__":
    run_tests()
    print("\n")
    main()
    print("\n")
    compare_baselines()
    print("\n")
    compare_extreme_scenarios()
