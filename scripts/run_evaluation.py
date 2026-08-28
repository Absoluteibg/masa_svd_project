"""
run_evaluation.py  —  Full benchmark evaluation on TruthfulQA.

Saves results CSV + prints metrics table.
Usage:  python scripts/run_evaluation.py [--n 50] [--fast]
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import config
from src.pipeline import MASASVDPipeline
from evaluation.benchmarks import run_truthfulqa
from evaluation.plots import generate_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int,  default=config.DEFAULT_EVAL_N,
                        help="Number of TruthfulQA samples (default 50)")
    parser.add_argument("--fast", action="store_true",
                        help="Disable NLI (faster)")
    args = parser.parse_args()

    config.ensure_dirs()
    pipeline = MASASVDPipeline(use_nli=not args.fast)

    print(f"\n[Eval] Running on {args.n} TruthfulQA samples …\n")
    results = run_truthfulqa(pipeline, n=args.n, use_nli=not args.fast)

    m = results["metrics"]
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS — TruthfulQA")
    print("=" * 50)
    print(f"  Precision          : {m['precision']:.4f}")
    print(f"  Recall             : {m['recall']:.4f}")
    print(f"  F1 Score           : {m['f1_score']:.4f}")
    print(f"  Accuracy           : {m['accuracy']:.4f}")
    print(f"  AUC                : {m['auc']:.4f}")
    print(f"  Hallucination Rate : {m['hallucination_rate']:.4f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print(f"  N Samples          : {m['n_samples']}")
    print("=" * 50)
    print("\n  Use these numbers in Chapter 5 of your report.\n")

    # Generate plots
    csv_path = config.RESULTS_DIR / "truthfulqa_results.csv"
    if csv_path.exists():
        generate_all(csv_path)
        print(f"  Figures saved to: {config.RESULTS_DIR}/figures/\n")


if __name__ == "__main__":
    main()