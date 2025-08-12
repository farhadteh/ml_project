"""
Fast import utilities to minimize ML library loading times.

This module provides lazy loading and optimized import patterns for heavy ML libraries.
"""

import os
from typing import Any


# Set environment variables for faster imports (must be set before importing torch)
def setup_fast_environment():
    """Configure environment variables for faster ML library imports."""
    # Disable some PyTorch initialization overhead
    os.environ["OMP_NUM_THREADS"] = "1"  # Reduce OpenMP threads
    os.environ["MKL_NUM_THREADS"] = "1"  # Reduce Intel MKL threads
    os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Reduce NumExpr threads
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Reduce OpenBLAS threads

    # Disable CUDA initialization (since we're using CPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Optimize transformers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism warnings
    os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Prevent online model fetching during import
    os.environ["HF_HUB_OFFLINE"] = "1"  # Keep HuggingFace hub offline during import


# Call this before any torch/transformers imports
setup_fast_environment()


class LazyImporter:
    """Lazy importer that only loads modules when actually accessed."""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module: Any | None = None

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            print(f"Loading {self.module_name}...")
            self._module = __import__(self.module_name, fromlist=[""])
        return getattr(self._module, name)

    @property
    def module(self) -> Any:
        """Get the actual module, loading it if necessary."""
        if self._module is None:
            print(f"Loading {self.module_name}...")
            self._module = __import__(self.module_name, fromlist=[""])
        return self._module


# Create lazy importers
torch = LazyImporter("torch")
transformers = LazyImporter("transformers")
sentence_transformers = LazyImporter("sentence_transformers")


def quick_test():
    """Quick test function that doesn't import heavy libraries until needed."""
    print("✅ Fast imports module loaded successfully!")
    print("Use torch.module, transformers.module, etc. when you need the actual libraries")
    return True


if __name__ == "__main__":
    quick_test()
