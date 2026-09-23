"""
Evaluation metrics for PRISM-Phish.

Comprehensive metric computation for phishing detection evaluation.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    prefix: str = "",
) -> dict:
    """
    Compute the full suite of classification metrics.

    Args:
        y_true: Ground truth labels (0/1).
        y_pred: Predicted labels (0/1).
        y_prob: Predicted probability of positive class (phishing).
        prefix: Optional prefix for metric keys.

    Returns:
        Dict with all metrics.
    """
    metrics = {}
    p = f"{prefix}_" if prefix else ""

    # Basic metrics
    metrics[f"{p}accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics[f"{p}balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics[f"{p}precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics[f"{p}recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics[f"{p}f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics[f"{p}f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics[f"{p}mcc"] = float(matthews_corrcoef(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics[f"{p}tp"] = int(tp)
    metrics[f"{p}fp"] = int(fp)
    metrics[f"{p}tn"] = int(tn)
    metrics[f"{p}fn"] = int(fn)

    # Derived rates
    metrics[f"{p}specificity"] = float(tn / max(tn + fp, 1))
    metrics[f"{p}fpr"] = float(fp / max(fp + tn, 1))
    metrics[f"{p}fnr"] = float(fn / max(fn + tp, 1))

    # Probability-based metrics
    if y_prob is not None:
        try:
            metrics[f"{p}roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics[f"{p}roc_auc"] = float("nan")

        try:
            metrics[f"{p}pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics[f"{p}pr_auc"] = float("nan")

        metrics[f"{p}brier_score"] = float(brier_score_loss(y_true, y_prob))

        # Expected Calibration Error
        metrics[f"{p}ece"] = float(expected_calibration_error(y_true, y_prob))

    return metrics


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well predicted probabilities match observed frequencies.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue

        bin_accuracy = y_true[mask].mean()
        bin_confidence = y_prob[mask].mean()
        bin_weight = mask.sum() / len(y_true)
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def format_metrics_table(metrics: dict, title: str = "Metrics") -> str:
    """Format metrics as a readable table."""
    lines = [f"\n{'='*50}", f"  {title}", f"{'='*50}"]
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"  {key:<30} {value:.4f}")
        else:
            lines.append(f"  {key:<30} {value}")
    lines.append(f"{'='*50}")
    return "\n".join(lines)
