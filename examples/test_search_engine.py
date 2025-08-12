#!/usr/bin/env python3
"""
Simple script to test the search engine functionality.
Run this to verify everything works before using the notebook.
"""

import sys

sys.path.append("../src")

from search_engine import CONFIG, DOCS, create_search_context, run_tests, search


def main():
    print("🚀 SEARCH ENGINE QUICK TEST")
    print("=" * 40)

    # Create search context
    print("Building search context...")
    ctx = create_search_context(DOCS, CONFIG)
    print("✅ Context created!")

    # Test basic queries
    test_queries = ["resume", "cv", "A4 minimal template", "gold logo modern"]

    print("\n🔍 TESTING BASIC QUERIES:")
    for query in test_queries:
        results = search(query, ctx, CONFIG, k=3)
        print(f"\nQuery: '{query}' → {len(results)} results")
        for i, (doc_id, score) in enumerate(results):
            doc = next(d for d in DOCS if d["id"] == doc_id)
            print(f"  {i+1}. [{doc_id}] {score:.3f} - {doc['title']}")

    print("\n🧪 RUNNING COMPREHENSIVE TESTS:")
    run_tests()

    print("\n✅ All tests passed! The search engine is ready to use.")
    print("📓 Open notebooks/search_engine_demo.ipynb for interactive demo!")


if __name__ == "__main__":
    main()
