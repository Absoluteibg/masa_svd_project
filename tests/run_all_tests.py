"""
run_all_tests.py  —  Runs the full test suite and prints a summary.

Usage:  python tests/run_all_tests.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

def main():
    print("\n" + "="*60)
    print("  MASA-SVD — Full Test Suite")
    print("="*60 + "\n")

    exit_code = pytest.main([
        "tests/",
        "-v",
        "--tb=short",
        "--no-header",
    ])

    print("\n" + "="*60)
    if exit_code == 0:
        print("  ✅ ALL TESTS PASSED")
    else:
        print("  ❌ SOME TESTS FAILED — check output above")
    print("="*60 + "\n")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
    