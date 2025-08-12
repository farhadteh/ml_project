#!/usr/bin/env python3
"""
Setup validation script to verify the search engine dependencies are correctly installed.
Run this after following the README setup instructions.
"""

import sys
from importlib import import_module


def check_import(module_name, description=""):
    """Check if a module can be imported."""
    try:
        import_module(module_name)
        print(f"✅ {module_name:<20} {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name:<20} {description} - ERROR: {e}")
        return False


def main():
    print("🔍 SEARCH ENGINE SETUP VALIDATION")
    print("=" * 50)

    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"✅ Python {python_version:<12} (Required: 3.12+)")

    print("\n📦 CHECKING CORE DEPENDENCIES:")

    # Core dependencies for search engine
    core_deps = [
        ("rank_bm25", "BM25 algorithm implementation"),
        ("sklearn", "Scikit-learn for TF-IDF"),
        ("spacy", "spaCy for tokenization"),
        ("numpy", "Numerical computing"),
        ("pandas", "Data manipulation"),
        ("math", "Python standard library"),
        ("collections", "Python standard library"),
        ("typing", "Type annotations"),
    ]

    core_success = all(check_import(mod, desc) for mod, desc in core_deps)

    print("\n🧪 CHECKING TEST DEPENDENCIES:")
    test_deps = [
        ("pytest", "Testing framework"),
    ]

    test_success = all(check_import(mod, desc) for mod, desc in test_deps)

    print("\n📓 CHECKING OPTIONAL DEPENDENCIES:")
    optional_deps = [
        ("jupyter", "Jupyter notebooks"),
        ("matplotlib", "Plotting library"),
        ("IPython", "Interactive Python"),
    ]

    optional_success = sum(check_import(mod, desc) for mod, desc in optional_deps)

    print("\n📊 SUMMARY:")
    print(f"Core dependencies: {'✅ All working' if core_success else '❌ Some missing'}")
    print(f"Test dependencies: {'✅ All working' if test_success else '❌ Some missing'}")
    print(f"Optional dependencies: {optional_success}/{len(optional_deps)} working")

    if core_success and test_success:
        print("\n🎉 SETUP COMPLETE! You can now:")
        print("   1. Run: cd examples && uv run python test_search_engine.py")
        print("   2. Run: uv run pytest")
        print("   3. Run: uv run jupyter lab (then open notebooks/search_engine_demo.ipynb)")
        return True
    else:
        print("\n⚠️  SETUP INCOMPLETE. Please run:")
        print("   uv sync")
        print("   uv sync --group dev")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
