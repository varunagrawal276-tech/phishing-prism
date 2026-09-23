import numpy as np
import pytest

from src.evaluation.cross_source import compute_loso_summary
from src.evaluation.metrics import compute_all_metrics


def test_compute_all_metrics():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.4])

    metrics = compute_all_metrics(y_true, y_pred, y_prob, prefix="test")
    assert "test_accuracy" in metrics
    assert "test_f1" in metrics
    assert "test_roc_auc" in metrics
    assert "test_pr_auc" in metrics
    assert metrics["test_accuracy"] == 0.8
    assert 0.0 <= metrics["test_roc_auc"] <= 1.0


def test_compute_loso_summary():
    loso_results = {
        "source_a": {"test_roc_auc": 0.95, "test_f1": 0.90, "test_pr_auc": 0.94},
        "source_b": {"test_roc_auc": 0.85, "test_f1": 0.80, "test_pr_auc": 0.84},
    }
    summary = compute_loso_summary(loso_results)
    assert "test_roc_auc_mean" in summary
    assert "test_f1_mean" in summary
    assert np.isclose(summary["test_roc_auc_mean"], 0.90)
    assert np.isclose(summary["test_f1_mean"], 0.85)

