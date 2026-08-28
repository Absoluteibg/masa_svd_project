"""
run_ablation.py  —  Ablation study: effect of SVD k and agent weights.

Runs evaluation on a small sample for each configuration and
saves a comparison CSV + bar chart.

Usage:  python scripts/run_ablation.py [--n 20]
"""
import sys, os, argparse, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm
from src.config import config
from src.pipeline import MASASVDPipeline
from evaluation.benchmarks import load_truthfulqa
from evaluation.metrics import full_report
from evaluation.plots import plot_ablation_k


def evaluate_config(pipeline: MASASVDPipeline,
                    items: list, use_nli: bool = False) -> dict:
    """Evaluate a configured pipeline on given items. Returns metrics dict."""
    y_true, y_pred, y_scores = [], [], []

    for item in items:
        q, cor, inc = item["question"], item["correct"], item["incorrect"]

        r_c = pipeline.analyze(q, cor, use_nli=use_nli)
        time.sleep(0.2)
        r_w = pipeline.analyze(q, inc, use_nli=use_nli)
        time.sleep(0.2)

        y_true  += [False,  True]
        y_pred  += [r_c["verdict"] == "HALLUCINATED",
                    r_w["verdict"] == "HALLUCINATED"]
        y_scores+= [r_c["final_risk_score"], r_w["final_risk_score"]]

    return full_report(y_true, y_pred, y_scores)


def ablation_k(items: list, k_values: list) -> list:
    """Test different SVD k values."""
    print("\n[Ablation] SVD k values:", k_values)
    rows = []
    f1s  = []
    for k in k_values:
        print(f"  Testing k={k} …")
        pipeline = MASASVDPipeline(svd_k=k, use_nli=False)
        m = evaluate_config(pipeline, items, use_nli=False)
        row = {"k": k, **m}
        rows.append(row)
        f1s.append(m["f1_score"])
        print(f"    k={k}: F1={m['f1_score']:.4f}  AUC={m['auc']:.4f}")

    plot_ablation_k(k_values, f1s)
    return rows


def ablation_weights(items: list, configs: list) -> list:
    """Test different agent weight combinations."""
    print("\n[Ablation] Agent weight combinations:")
    rows = []
    for α, β, γ in configs:
        if abs(α + β + γ - 1.0) > 1e-6:
            continue
        label = f"α={α} β={β} γ={γ}"
        print(f"  Testing {label} …")
        pipeline = MASASVDPipeline(
            alpha=α, beta=β, gamma=γ, use_nli=False
        )
        m = evaluate_config(pipeline, items, use_nli=False)
        rows.append({"config": label, "alpha": α, "beta": β, "gamma": γ, **m})
        print(f"    {label}: F1={m['f1_score']:.4f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20,
                        help="Samples for ablation (default 20)")
    args = parser.parse_args()

    config.ensure_dirs()
    items = load_truthfulqa(args.n)

    # ── K Ablation ────────────────────────────────────────────────────────
    k_results = ablation_k(items, k_values=[1, 3, 5, 7, 10])

    k_csv = config.RESULTS_DIR / "ablation_k.csv"
    with open(k_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=k_results[0].keys())
        writer.writeheader()
        writer.writerows(k_results)
    print(f"\n[Ablation] k-ablation saved → {k_csv}")

    # ── Weight Ablation ───────────────────────────────────────────────────
    weight_configs = [
        (0.40, 0.35, 0.25),   # default
        (0.50, 0.30, 0.20),
        (0.30, 0.50, 0.20),
        (0.33, 0.33, 0.34),   # equal weights
        (0.60, 0.25, 0.15),   # SVD-heavy
        (0.20, 0.60, 0.20),   # FVA-heavy
    ]
    w_results = ablation_weights(items, weight_configs)

    w_csv = config.RESULTS_DIR / "ablation_weights.csv"
    with open(w_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=w_results[0].keys())
        writer.writeheader()
        writer.writerows(w_results)
    print(f"[Ablation] weight-ablation saved → {w_csv}")

    print("\n[Ablation] Done. Use these tables in your Chapter 5 ablation section.")


if __name__ == "__main__":
    main()