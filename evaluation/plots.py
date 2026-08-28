"""
plots.py  —  Generate report-quality figures from evaluation results.

All figures saved as high-resolution PNG in results/figures/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Headless rendering (no display required)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import config


def _fig_dir() -> Path:
    d = config.RESULTS_DIR / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_sds_distribution(df: pd.DataFrame) -> Path:
    """Histogram of SDS scores for correct vs hallucinated responses."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["correct_sds"], bins=20, alpha=0.65,
            label="Correct Responses",     color="#2ecc71", edgecolor="white")
    ax.hist(df["wrong_sds"],   bins=20, alpha=0.65,
            label="Hallucinated Responses", color="#e74c3c", edgecolor="white")
    ax.axvline(x=0.45, color="navy", linestyle="--", linewidth=1.5,
               label="Threshold (τ = 0.45)")
    ax.set_xlabel("Subspace Divergence Score (SDS)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("SDS Distribution: Correct vs. Hallucinated Responses",
                 fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    path = _fig_dir() / "fig1_sds_distribution.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved → {path}")
    return path


def plot_risk_boxplot(df: pd.DataFrame) -> Path:
    """Box plot of final risk scores: correct vs hallucinated."""
    fig, ax = plt.subplots(figsize=(7, 5))
    data   = [df["correct_risk"].tolist(), df["wrong_risk"].tolist()]
    labels = ["Correct Responses", "Hallucinated Responses"]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})
    colors = ["#2ecc71", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Final Risk Score", fontsize=12)
    ax.set_title("Risk Score Distribution by Response Type",
                 fontsize=13, fontweight="bold")
    ax.axhline(y=0.45, color="navy", linestyle="--",
               linewidth=1.5, label="Threshold")
    ax.legend()
    fig.tight_layout()
    path = _fig_dir() / "fig2_risk_boxplot.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved → {path}")
    return path


def plot_ablation_k(k_values: list, f1_values: list) -> Path:
    """Bar chart for SVD k-ablation study results."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(k) for k in k_values], f1_values,
                  color="#3498db", alpha=0.8, edgecolor="white")
    best_idx = int(np.argmax(f1_values))
    bars[best_idx].set_color("#e67e22")
    ax.set_xlabel("Number of SVD Components (k)", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Ablation Study: Effect of SVD Components (k) on F1",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, f1_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", fontsize=10)
    fig.tight_layout()
    path = _fig_dir() / "fig3_ablation_k.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved → {path}")
    return path


def generate_all(results_csv: str | Path) -> None:
    """Load CSV from benchmark run and generate all figures."""
    df = pd.read_csv(results_csv)
    plot_sds_distribution(df)
    plot_risk_boxplot(df)
    print("[Plot] All figures saved to results/figures/")