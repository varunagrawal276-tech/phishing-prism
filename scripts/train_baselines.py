#!/usr/bin/env python3
"""
Train Classical ML Baselines — Stage 3

Trains:
1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVC (Calibrated)
3. TF-IDF + XGBoost Classifier

Evaluates on in-domain test set (ROC-AUC, PR-AUC, F1, Precision, Recall, FPR@99% Recall).
Saves models and metrics to artifacts/baselines/.

Usage:
    python scripts/train_baselines.py [--models lr,svc,xgb] [--seed 42]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_config
from src.evaluation.metrics import compute_all_metrics
from src.models.classical import (
    build_tfidf_features,
    train_logistic_regression,
    train_linear_svm,
    train_xgboost,
)
from src.training.seeds import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_baselines")


def prepare_text(df: pd.DataFrame) -> list[str]:
    """Combine subject and body into single text string."""
    subj = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    return (subj + " " + body).tolist()


def main():
    parser = argparse.ArgumentParser(description="Train classical baseline models")
    parser.add_argument("--models", type=str, default="lr,svc,xgb", help="Comma-separated model names")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-features", type=int, default=30000, help="Max TF-IDF features per analyzer")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config("models.yaml")
    splits_dir = PROJECT_ROOT / "data" / "splits"
    output_dir = PROJECT_ROOT / "artifacts" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load splits
    logger.info("=== Loading data splits ===")
    train_path = splits_dir / "train.parquet"
    val_path = splits_dir / "val.parquet"
    test_path = splits_dir / "test.parquet"

    if not train_path.exists():
        logger.error(f"Train split not found at {train_path}. Run build_splits.py first.")
        sys.exit(1)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    logger.info(f"Loaded splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Prepare texts and labels
    train_texts = prepare_text(train_df)
    val_texts = prepare_text(val_df)
    test_texts = prepare_text(test_df)

    y_train = train_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    # 2. Build TF-IDF features
    logger.info("=== Extracting TF-IDF Features (Word + Char n-grams) ===")
    start_time = time.time()
    feature_dict, vectorizers = build_tfidf_features(
        train_texts=train_texts,
        val_texts=val_texts,
        test_texts=test_texts,
        max_features_word=args.max_features,
        max_features_char=args.max_features,
    )
    logger.info(f"Feature extraction completed in {time.time() - start_time:.1f}s")

    X_train = feature_dict["train"]
    X_val = feature_dict["val"]
    X_test = feature_dict["test"]

    # Save vectorizers
    joblib.dump(vectorizers, output_dir / "tfidf_vectorizers.joblib")

    # 3. Train models
    selected_models = [m.strip().lower() for m in args.models.split(",")]
    all_results = {}

    # Logistic Regression
    if "lr" in selected_models:
        logger.info("=== Training Logistic Regression ===")
        lr_result = train_logistic_regression(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
        )
        model = lr_result["model"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_all_metrics(y_test, y_pred, y_prob, prefix="test")
        logger.info(f"LR Test Results: ROC-AUC={metrics['test_roc_auc']:.4f}, PR-AUC={metrics['test_pr_auc']:.4f}, F1={metrics['test_f1']:.4f}")
        all_results["logistic_regression"] = {
            "metrics": metrics,
            "best_params": lr_result.get("best_params", {}),
            "training_time": lr_result.get("training_time", 0),
        }
        joblib.dump(model, output_dir / "logistic_regression.joblib")

    # Linear SVM
    if "svc" in selected_models or "svm" in selected_models:
        logger.info("=== Training Linear SVC ===")
        svm_result = train_linear_svm(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
        )
        model = svm_result["model"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = compute_all_metrics(y_test, y_pred, y_prob, prefix="test")
        logger.info(f"SVM Test Results: ROC-AUC={metrics.get('test_roc_auc', 0):.4f}, F1={metrics['test_f1']:.4f}")
        all_results["linear_svm"] = {
            "metrics": metrics,
            "best_params": svm_result.get("best_params", {}),
            "training_time": svm_result.get("training_time", 0),
        }
        joblib.dump(model, output_dir / "linear_svm.joblib")

    # XGBoost
    if "xgb" in selected_models or "xgboost" in selected_models:
        logger.info("=== Training XGBoost Classifier ===")
        xgb_result = train_xgboost(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
        )
        model = xgb_result["model"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_all_metrics(y_test, y_pred, y_prob, prefix="test")
        logger.info(f"XGBoost Test Results: ROC-AUC={metrics['test_roc_auc']:.4f}, PR-AUC={metrics['test_pr_auc']:.4f}, F1={metrics['test_f1']:.4f}")
        all_results["xgboost"] = {
            "metrics": metrics,
            "training_time": xgb_result.get("training_time", 0),
        }
        joblib.dump(model, output_dir / "xgboost.joblib")

    # Save summary report
    summary_path = output_dir / "baseline_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"All baseline results saved to {summary_path}")

    # Generate Markdown Summary Table
    md_lines = [
        "# Baseline Models Evaluation Results",
        "",
        "| Model | Test ROC-AUC | Test PR-AUC | Test F1 | Precision | Recall | Training Time |",
        "|---|---|---|---|---|---|---|",
    ]
    for model_name, res in all_results.items():
        m = res["metrics"]
        md_lines.append(
            f"| {model_name} | {m.get('test_roc_auc', 0):.4f} | {m.get('test_pr_auc', 0):.4f} | "
            f"{m.get('test_f1', 0):.4f} | {m.get('test_precision', 0):.4f} | {m.get('test_recall', 0):.4f} | "
            f"{res.get('training_time', 0):.1f}s |"
        )
    with open(output_dir / "baseline_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info("Markdown summary table saved to artifacts/baselines/baseline_summary.md")


if __name__ == "__main__":
    main()
