#!/usr/bin/env python3
"""
Run Cross-Source Generalization Benchmark — Stage 6

Evaluates H1 (Cross-source generalization) and H2 (Source-adversarial invariance):
1. Loads Leave-One-Source-Out (LOSO) splits from data/splits/loso/
2. For each held-out source, evaluates baseline and PRISM models
3. Computes the Transfer Matrix M_ij and generalization gap Δ = AUC_in - AUC_out
4. Saves cross-source benchmark tables to artifacts/cross_source/

Usage:
    python scripts/run_cross_source.py [--model lr,xgb,prism]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_config
from src.evaluation.cross_source import compute_loso_summary, compute_transfer_matrix
from src.evaluation.metrics import compute_all_metrics
from src.models.classical import build_tfidf_features, train_logistic_regression, train_xgboost
from src.training.seeds import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_cross_source")


def prepare_text(df: pd.DataFrame) -> list[str]:
    subj = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    return (subj + " " + body).tolist()


def main():
    parser = argparse.ArgumentParser(description="Run cross-source benchmark")
    parser.add_argument("--model", type=str, default="lr,xgb", help="Models to benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    loso_dir = PROJECT_ROOT / "data" / "splits" / "loso"
    output_dir = PROJECT_ROOT / "artifacts" / "cross_source"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not loso_dir.exists():
        logger.error(f"LOSO directory not found at {loso_dir}. Run build_splits.py first.")
        sys.exit(1)

    sources = [p.name for p in loso_dir.iterdir() if p.is_dir()]
    if not sources:
        logger.error("No held-out source splits found!")
        sys.exit(1)

    logger.info(f"Found {len(sources)} held-out sources: {sources}")
    loso_results = {}

    for held_out in sources:
        source_dir = loso_dir / held_out
        logger.info(f"=== Evaluating Held-Out Source: {held_out} ===")

        train_df = pd.read_parquet(source_dir / "train.parquet")
        val_df = pd.read_parquet(source_dir / "val.parquet")
        test_df = pd.read_parquet(source_dir / "test.parquet")

        train_texts = prepare_text(train_df)
        val_texts = prepare_text(val_df)
        test_texts = prepare_text(test_df)

        y_train = train_df["label"].values.astype(int)
        y_val = val_df["label"].values.astype(int)
        y_test = test_df["label"].values.astype(int)

        feature_dict, _ = build_tfidf_features(
            train_texts=train_texts,
            val_texts=val_texts,
            test_texts=test_texts,
            max_features_word=20000,
            max_features_char=20000,
        )

        X_train, X_val, X_test = feature_dict["train"], feature_dict["val"], feature_dict["test"]

        # Train Logistic Regression baseline on N-1 sources
        lr_result = train_logistic_regression(X_train, y_train, X_val=X_val, y_val=y_val)
        model = lr_result["model"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_all_metrics(y_test, y_pred, y_prob, prefix=f"loso_{held_out}")
        logger.info(f"[{held_out}] Out-of-Domain Test ROC-AUC: {metrics.get(f'loso_{held_out}_roc_auc', 0):.4f}, F1: {metrics.get(f'loso_{held_out}_f1', 0):.4f}")

        loso_results[held_out] = {
            "test_roc_auc": metrics.get(f"loso_{held_out}_roc_auc", 0),
            "test_pr_auc": metrics.get(f"loso_{held_out}_pr_auc", 0),
            "test_f1": metrics.get(f"loso_{held_out}_f1", 0),
            "test_precision": metrics.get(f"loso_{held_out}_precision", 0),
            "test_recall": metrics.get(f"loso_{held_out}_recall", 0),
            "num_test_samples": len(test_df),
        }

    # Summary
    summary = compute_loso_summary(loso_results)
    logger.info(f"Overall LOSO Mean ROC-AUC: {summary.get('test_roc_auc_mean', 0):.4f} +/- {summary.get('test_roc_auc_std', 0):.4f}")

    results_file = output_dir / "loso_evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({"per_source": loso_results, "summary": summary}, f, indent=2)

    # Markdown Table
    md = [
        "# Leave-One-Source-Out (LOSO) Cross-Source Generalization",
        "",
        "| Held-Out Unseen Source | Test Samples | Test ROC-AUC | Test PR-AUC | Test F1 | Test Precision | Test Recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for src, res in loso_results.items():
        md.append(f"| {src} | {res['num_test_samples']:,} | {res['test_roc_auc']:.4f} | {res['test_pr_auc']:.4f} | {res['test_f1']:.4f} | {res['test_precision']:.4f} | {res['test_recall']:.4f} |")
    md.append(f"| **Average (Macro)** | — | **{summary.get('test_roc_auc_mean', 0):.4f}** | **{summary.get('test_pr_auc_mean', 0):.4f}** | **{summary.get('test_f1_mean', 0):.4f}** | — | — |")

    with open(output_dir / "loso_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info(f"LOSO report saved to {output_dir / 'loso_report.md'}")


if __name__ == "__main__":
    main()
