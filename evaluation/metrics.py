"""
metrics.py  —  Classification metrics for hallucination detection.

All functions accept lists of ground-truth labels (bool) and
system predictions (bool or float risk scores).
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple


def confusion_matrix_values(
    y_true: List[bool], y_pred: List[bool]
) -> Tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN)."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    return tp, fp, tn, fn


def precision(y_true: List[bool], y_pred: List[bool]) -> float:
    tp, fp, _, _ = confusion_matrix_values(y_true, y_pred)
    return round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0


def recall(y_true: List[bool], y_pred: List[bool]) -> float:
    tp, _, _, fn = confusion_matrix_values(y_true, y_pred)
    return round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0


def f1_score(y_true: List[bool], y_pred: List[bool]) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0


def accuracy(y_true: List[bool], y_pred: List[bool]) -> float:
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return round(correct / len(y_true), 4) if y_true else 0.0


def roc_auc(y_true: List[bool], y_scores: List[float]) -> float:
    """Compute rank-based AUC, awarding half credit to tied scores."""
    pairs = sorted(zip(y_scores, y_true), key=lambda item: item[0], reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = fp = 0
    auc = 0.0

    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        group_end = index
        positives = negatives = 0
        while group_end < len(pairs) and pairs[group_end][0] == score:
            if pairs[group_end][1]:
                positives += 1
            else:
                negatives += 1
            group_end += 1

        # Each tied positive beats preceding negatives and receives half credit
        # against negatives in its own tie group.
        auc += positives * fp + 0.5 * positives * negatives
        tp += positives
        fp += negatives
        index = group_end

    return round(auc / (n_pos * n_neg), 4)


def hallucination_rate(y_true: List[bool], y_pred: List[bool]) -> float:
    """Fraction of real hallucinations that were MISSED (false negative rate)."""
    _, _, _, fn = confusion_matrix_values(y_true, y_pred)
    total_positives = sum(y_true)
    return round(fn / total_positives, 4) if total_positives > 0 else 0.0


def full_report(y_true: List[bool], y_pred: List[bool],
                y_scores: List[float]) -> dict:
    """Return all metrics in one dict."""
    tp, fp, tn, fn = confusion_matrix_values(y_true, y_pred)
    return {
        "precision":          precision(y_true, y_pred),
        "recall":             recall(y_true, y_pred),
        "f1_score":           f1_score(y_true, y_pred),
        "accuracy":           accuracy(y_true, y_pred),
        "auc":                roc_auc(y_true, y_scores),
        "hallucination_rate": hallucination_rate(y_true, y_pred),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_samples": len(y_true),
    }
