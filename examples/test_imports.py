#!/usr/bin/env python3
"""
Test script to verify all ML packages can be imported successfully.

This script tests all the heavy ML libraries we installed to ensure
they work correctly after resolving the torchvision dependency issue.
"""

import sys
import time


def test_package_import(package_name: str, import_statement: str):
    """Test importing a package and measure time."""
    print(f"Testing {package_name}...", end=" ")
    start_time = time.time()

    try:
        exec(import_statement)
        duration = time.time() - start_time
        print(f"✅ Success ({duration:.2f}s)")
        return True
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Failed ({duration:.2f}s): {e}")
        return False


def main():
    """Test all ML package imports."""
    print("🧪 Testing ML Package Imports")
    print("=" * 50)

    # Core packages
    test_packages = [
        ("NumPy", "import numpy as np; print(f'  NumPy {np.__version__}')"),
        ("Pandas", "import pandas as pd; print(f'  Pandas {pd.__version__}')"),
        ("Scikit-learn", "import sklearn; print(f'  Scikit-learn {sklearn.__version__}')"),
        ("XGBoost", "import xgboost as xgb; print(f'  XGBoost {xgb.__version__}')"),
    ]

    # Deep learning packages
    deep_learning_packages = [
        ("PyTorch", "import torch; print(f'  PyTorch {torch.__version__}')"),
        ("TorchVision", "import torchvision; print(f'  TorchVision {torchvision.__version__}')"),
        (
            "Transformers",
            "import transformers; print(f'  Transformers {transformers.__version__}')",
        ),
        (
            "Sentence Transformers",
            "import sentence_transformers; "
            "print(f'  Sentence Transformers {sentence_transformers.__version__}')",
        ),
        (
            "HuggingFace Hub",
            "import huggingface_hub; print(f'  HuggingFace Hub {huggingface_hub.__version__}')",
        ),
        ("Safetensors", "import safetensors; print(f'  Safetensors {safetensors.__version__}')"),
        ("spaCy", "import spacy; print(f'  spaCy {spacy.__version__}')"),
        ("OpenAI CLIP", "import clip; print(f'  OpenAI CLIP imported successfully')"),
    ]

    # Other packages
    other_packages = [
        ("BM25", "from rank_bm25 import BM25Okapi; print('  BM25 imported successfully')"),
        ("LightGBM", "import lightgbm as lgb; print(f'  LightGBM {lgb.__version__}')"),
        ("Streamlit", "import streamlit as st; print(f'  Streamlit {st.__version__}')"),
        ("Redis", "import redis; print(f'  Redis {redis.__version__}')"),
        ("SciPy", "import scipy; print(f'  SciPy {scipy.__version__}')"),
        ("tqdm", "import tqdm; print(f'  tqdm {tqdm.__version__}')"),
    ]

    # Test all packages
    all_packages = test_packages + deep_learning_packages + other_packages

    total_start = time.time()
    success_count = 0

    for package_name, import_stmt in all_packages:
        if test_package_import(package_name, import_stmt):
            success_count += 1

    total_time = time.time() - total_start

    print("\n" + "=" * 50)
    print(f"🎯 Results: {success_count}/{len(all_packages)} packages imported successfully")
    print(f"⏱️  Total time: {total_time:.2f}s")

    if success_count == len(all_packages):
        print("🎉 All packages working correctly!")
        return True
    else:
        print(f"⚠️  {len(all_packages) - success_count} packages failed to import")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
