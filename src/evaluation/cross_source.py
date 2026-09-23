"""
Cross-source evaluation for PRISM-Phish.

Implements:
- Leave-One-Source-Out (LOSO) evaluation metrics
- Pairwise Source-to-Source Transfer Matrix (M_ij)
- Cross-Domain Robustness Gap: Δ = Metric_in - Metric_out
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

logger = logging.getLogger(__name__)


def compute_transfer_matrix(
    models: Dict[str, any],
    eval_data: Dict[str, Tuple[any, np.ndarray]],
    metric_fn: Optional[callable] = None,
) -> pd.DataFrame:
    """
    Compute pairwise transfer matrix M_ij.
    M[i, j] is the performance of model trained on source i tested on source j.

    Args:
        models: Dict mapping source_name -> trained model
        eval_data: Dict mapping source_name -> (X_test, y_test)
        metric_fn: Metric function (defaults to ROC-AUC)

    Returns:
        pd.DataFrame of shape (n_sources, n_sources)
    """
    sources = sorted(list(models.keys()))
    matrix = pd.DataFrame(index=sources, columns=sources, dtype=float)

    for src_i in sources:
        model = models[src_i]
        for src_j in sources:
            if src_j not in eval_data:
                continue
            X_j, y_j = eval_data[src_j]
            if len(np.unique(y_j)) < 2:
                matrix.loc[src_i, src_j] = np.nan
                continue

            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_j)[:, 1]
                    score = roc_auc_score(y_j, probs)
                else:
                    preds = model.predict(X_j)
                    score = f1_score(y_j, preds, zero_division=0)
                matrix.loc[src_i, src_j] = round(float(score), 4)
            except Exception as e:
                logger.warning(f"Failed evaluation M[{src_i}, {src_j}]: {e}")
                matrix.loc[src_i, src_j] = np.nan

    return matrix


def compute_loso_summary(loso_results: Dict[str, dict]) -> dict:
    """
    Summarize Leave-One-Source-Out results across all held-out sources.

    Args:
        loso_results: Dict mapping held-out source -> test metrics dict

    Returns:
        Summary dict with mean, std, min, max for each metric across sources.
    """
    metrics_summary = {}
    sample_metrics = next(iter(loso_results.values()))

    for metric_name in sample_metrics.keys():
        if isinstance(sample_metrics[metric_name], (int, float)):
            values = [res[metric_name] for res in loso_results.values() if metric_name in res and not np.isnan(res[metric_name])]
            if values:
                metrics_summary[f"{metric_name}_mean"] = round(float(np.mean(values)), 4)
                metrics_summary[f"{metric_name}_std"] = round(float(np.std(values)), 4)
                metrics_summary[f"{metric_name}_min"] = round(float(np.min(values)), 4)
                metrics_summary[f"{metric_name}_max"] = round(float(np.max(values)), 4)

    return metrics_summary


def compute_generalization_gap(in_domain_score: float, out_domain_score: float) -> float:
    """Compute drop in performance when moving to unseen domain."""
    return round(float(in_domain_score - out_domain_score), 4)
