"""Focused regression tests for evaluation metrics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.metrics import roc_auc


def test_auc_assigns_half_credit_to_ties():
    """An indistinguishable positive and negative should have AUC of 0.5."""
    assert roc_auc([False, True], [0.5, 0.5]) == 0.5


def test_auc_is_one_for_perfect_separation():
    assert roc_auc([False, True], [0.1, 0.9]) == 1.0
