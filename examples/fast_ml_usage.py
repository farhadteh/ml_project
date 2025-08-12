#!/usr/bin/env python3
"""
Example: Fast ML Library Usage Patterns

This demonstrates how to minimize import overhead for PyTorch and transformers
while maintaining full functionality when needed.
"""

import os
import sys
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Method 1: Environment-optimized direct imports
def setup_fast_environment():
    """Configure environment variables for faster imports."""
    os.environ.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )


# Method 2: Lazy loading pattern
class LazyTorch:
    def __init__(self):
        self._torch = None

    @property
    def torch(self):
        if self._torch is None:
            print("Loading PyTorch...")
            start = time.time()
            import torch

            self._torch = torch
            print(f"PyTorch loaded in {time.time() - start:.2f}s")
        return self._torch


# Method 3: Fast ML operations without heavy imports
def fast_text_similarity(text1: str, text2: str) -> float:
    """Fast text similarity using simple token overlap (no transformers needed)."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    return len(intersection) / len(union) if union else 0.0


def fast_ranking_demo():
    """Demonstrate fast ranking without heavy imports."""
    documents = [
        {"id": "1", "text": "Python machine learning tutorial", "clicks": 150},
        {"id": "2", "text": "Java programming guide", "clicks": 80},
        {"id": "3", "text": "PyTorch deep learning course", "clicks": 200},
        {"id": "4", "text": "Machine learning with Python", "clicks": 120},
    ]

    query = "Python machine learning"

    # Fast similarity scoring
    scores = []
    for doc in documents:
        similarity = fast_text_similarity(query, doc["text"])
        popularity = doc["clicks"] / 100.0  # Simple popularity score
        final_score = similarity * 0.7 + popularity * 0.3
        scores.append((doc["id"], final_score, doc["text"]))

    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)

    print("🚀 Fast Ranking Results:")
    for doc_id, score, text in scores:
        print(f"  {doc_id}: {score:.3f} - {text}")

    return scores


# Method 4: Only load transformers when actually needed for embeddings
def semantic_search_when_needed(query: str, documents: list[str], use_semantic: bool = False):
    """Only import transformers if semantic search is requested."""

    if not use_semantic:
        print("Using fast lexical search...")
        # Use simple keyword matching
        results = []
        query_words = set(query.lower().split())

        for i, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            score = len(query_words.intersection(doc_words)) / len(query_words)
            results.append((i, score, doc))

        return sorted(results, key=lambda x: x[1], reverse=True)

    else:
        print("Loading sentence transformers for semantic search...")
        start_time = time.time()

        # Only now do we import the heavy library
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")

        # Compute embeddings
        query_embedding = model.encode([query])
        doc_embeddings = model.encode(documents)

        # Compute similarities
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

        results = [(i, similarities[i], doc) for i, doc in enumerate(documents)]
        return sorted(results, key=lambda x: x[1], reverse=True)


def main():
    """Run fast ML usage examples."""
    setup_fast_environment()

    print("=" * 60)
    print("Fast ML Library Usage Examples")
    print("=" * 60)

    # Example 1: Fast ranking without imports
    print("\n1. Fast Ranking (No Heavy Imports)")
    print("-" * 40)
    start = time.time()
    fast_ranking_demo()
    print(f"⚡ Completed in {time.time() - start:.3f}s")

    # Example 2: Conditional semantic search
    print("\n2. Conditional Semantic Search")
    print("-" * 40)
    docs = [
        "Python programming for beginners",
        "Advanced machine learning algorithms",
        "Data science with pandas",
        "Deep learning neural networks",
    ]

    # Fast search first
    print("\nFast lexical search:")
    start = time.time()
    results = semantic_search_when_needed("Python programming", docs, use_semantic=False)
    for _i, score, doc in results[:2]:
        print(f"  {score:.3f}: {doc}")
    print(f"⚡ Completed in {time.time() - start:.3f}s")

    # Example 3: Lazy loading pattern
    print("\n3. Lazy Loading PyTorch")
    print("-" * 40)
    lazy_torch = LazyTorch()
    print("LazyTorch created (no import yet)")

    # Only loads when actually accessed
    print("Accessing torch.version...")
    start = time.time()
    version = lazy_torch.torch.__version__
    print(f"PyTorch version: {version}")
    print(f"⚡ First access took {time.time() - start:.3f}s")

    # Second access is instant
    print("Second access to torch...")
    start = time.time()
    device = lazy_torch.torch.device("cpu")
    print(f"Device: {device}")
    print(f"⚡ Second access took {time.time() - start:.3f}s")


if __name__ == "__main__":
    main()
