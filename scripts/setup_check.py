"""
setup_check.py  —  Run this FIRST before anything else.
Verifies your environment is correctly set up.

Usage:  python scripts/setup_check.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import importlib

def check(label: str, fn):
    try:
        result = fn()
        print(f"  ✅  {label}" + (f" — {result}" if result else ""))
        return True
    except Exception as e:
        print(f"  ❌  {label} — ERROR: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("  MASA-SVD Environment Check")
    print("="*60)
    
    all_ok = True

    # ── Python version ────────────────────────────────────────────────────
    print("\n[1] Python")
    v = sys.version_info
    ok = check("Python version",
               lambda: f"{v.major}.{v.minor}.{v.micro} ✓"
                       if v >= (3, 9) else (_ for _ in ()).throw(
                           RuntimeError(f"Need ≥ 3.9, got {v.major}.{v.minor}")))
    all_ok &= ok

    # ── Package imports ───────────────────────────────────────────────────
    print("\n[2] Package Imports")

packages = [
    "numpy",
    "scipy",
    "sentence_transformers",
    "openai",
    "fastapi",
    "torch",
    "transformers",
    "datasets",
    "sklearn",
    "pandas",
    "matplotlib",
    "tqdm",
]

for name in packages:
    ok = check(
        name,
        lambda n=name: importlib.import_module(n).__version__
    )
    all_ok = ok

    # ── Project imports ───────────────────────────────────────────────────

modules = [
    "src.config",
    "src.core.embeddings",
    "src.core.svd_engine",
    "src.agents.factual_agent",
    "src.agents.semantic_agent",
    "src.agents.calibration_agent",
    "src.pipeline",
]

for name in modules:
    ok = check(
        name,
        lambda n=name: importlib.import_module(n)
    )
    all_ok &= ok

    # ── .env and API key ─────────────────────────────────────────────────
    print("\n[4] Configuration")
    from src.config import config
    ok = check(".env file found",
               lambda: "Yes" if (config.PROJECT_ROOT / ".env").exists()
                       else (_ for _ in ()).throw(FileNotFoundError(
                           f"Create {config.PROJECT_ROOT}/.env from .env.example")))
    all_ok &= ok

    check("OpenAI API key",
          lambda: "SET ✓" if config.has_openai_key()
                  else (_ for _ in ()).throw(
                      ValueError("Not set — FVA will use fallback (0.5)")))

    # ── Quick SVD test ────────────────────────────────────────────────────
    print("\n[5] Quick Functional Test")
    def _svd_test():
        from src.core.embeddings import embed_texts
        from src.core.svd_engine import SVDSubspaceEngine
        e   = SVDSubspaceEngine(k=3)
        sds = e.compute_sds(["What is AI?"],
                            ["AI stands for Artificial Intelligence."])
        assert 0.0 <= sds <= 1.0, f"SDS out of range: {sds}"
        return f"SDS = {sds}"

    ok = check("SVD pipeline smoke test", _svd_test)
    all_ok &= ok

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    if all_ok:
        print("  ✅  ALL CHECKS PASSED — you're ready to run the project!")
    else:
        print("  ⚠️  SOME CHECKS FAILED — fix the ❌ items above first.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()