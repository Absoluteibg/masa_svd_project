"""
benchmarks.py  —  Runs MASA-SVD against TruthfulQA.

For each question in the dataset we analyze:
  - The CORRECT answer  → expect VERIFIED
  - One WRONG answer    → expect HALLUCINATED

Results are saved to results/truthfulqa_results.csv
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

from src.config import config
from src.pipeline import MASASVDPipeline
from evaluation.metrics import full_report

log = logging.getLogger(__name__)


def load_truthfulqa(n: int) -> List[Dict]:
    """Download and return first n items from TruthfulQA (generation split)."""
    from datasets import load_dataset
    print(f"[Benchmark] Loading TruthfulQA ({n} samples) …")
    ds = load_dataset("truthful_qa", "generation",
                      trust_remote_code=True, split="validation")
    items = []
    for row in list(ds)[:n]:
        if row.get("best_answer") and row.get("incorrect_answers"):
            items.append({
                "question":   row["question"],
                "correct":    row["best_answer"],
                "incorrect":  row["incorrect_answers"][0],
            })
    print(f"[Benchmark] Loaded {len(items)} items.")
    return items


def run_truthfulqa(
    pipeline: MASASVDPipeline,
    n:        int  = config.DEFAULT_EVAL_N,
    use_nli:  bool = True,
    delay:    float = 0.3,   # seconds between API calls (rate-limit guard)
) -> Dict:
    """
    Evaluate pipeline on TruthfulQA.

    Parameters
    ----------
    pipeline : Initialised MASASVDPipeline.
    n        : Number of questions to evaluate.
    use_nli  : Whether to enable NLI in SCA.
    delay    : Sleep between items to avoid API rate limits.

    Returns
    -------
    dict with metrics + raw_results list.
    """
    config.ensure_dirs()
    items = load_truthfulqa(n)

    y_true:   List[bool]  = []  # True = hallucinated
    y_pred:   List[bool]  = []  # True = predicted hallucinated
    y_scores: List[float] = []  # risk scores for AUC

    rows = []   # For CSV export

    for item in tqdm(items, desc="Evaluating"):
        q   = item["question"]
        cor = item["correct"]
        inc = item["incorrect"]

        r_correct = pipeline.analyze(q, cor,   use_nli=use_nli)
        time.sleep(delay)
        r_wrong   = pipeline.analyze(q, inc,   use_nli=use_nli)
        time.sleep(delay)

        # Correct answer should be VERIFIED (not hallucinated = False)
        y_true.append(False)
        y_pred.append(r_correct["verdict"] == "HALLUCINATED")
        y_scores.append(r_correct["final_risk_score"])

        # Wrong answer should be HALLUCINATED (is hallucinated = True)
        y_true.append(True)
        y_pred.append(r_wrong["verdict"] == "HALLUCINATED")
        y_scores.append(r_wrong["final_risk_score"])

        rows.append({
            "question":        q,
            "correct_answer":  cor,
            "wrong_answer":    inc,
            "correct_verdict": r_correct["verdict"],
            "correct_risk":    r_correct["final_risk_score"],
            "correct_sds":     r_correct["breakdown"]["sds"],
            "wrong_verdict":   r_wrong["verdict"],
            "wrong_risk":      r_wrong["final_risk_score"],
            "wrong_sds":       r_wrong["breakdown"]["sds"],
        })

    # ── Save CSV ─────────────────────────────────────────────────────────────
    out_path = config.RESULTS_DIR / "truthfulqa_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Benchmark] Results saved → {out_path}")

    metrics = full_report(y_true, y_pred, y_scores)
    return {"metrics": metrics, "raw_results": rows}