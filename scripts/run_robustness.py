#!/usr/bin/env python3
"""
Run Adversarial Perturbation & Robustness Benchmark — Stage 7

Tests Hypothesis H3:
"Models trained with perturbation consistency maintain higher detection rates under
homoglyph, whitespace, URL obfuscation, and prompt-injection perturbations."

Usage:
    python scripts/run_robustness.py [--model lr,xgb,prism] [--severity 0.1]
"""

import argparse
import json
import logging
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
from src.robustness.perturbations import apply_perturbations
from src.training.seeds import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_robustness")


def prepare_text(df: pd.DataFrame) -> list[str]:
    subj = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    return (subj + " " + body).tolist()


def main():
    parser = argparse.ArgumentParser(description="Run adversarial robustness benchmark")
    parser.add_argument("--severity", type=float, default=0.1, help="Perturbation intensity (0.05 to 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nrows", type=int, default=2000, help="Max test samples for perturbation testing")
    args = parser.parse_args()

    set_seed(args.seed)
    test_path = PROJECT_ROOT / "data" / "splits" / "test.parquet"
    models_dir = PROJECT_ROOT / "artifacts" / "baselines"
    output_dir = PROJECT_ROOT / "artifacts" / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_path.exists():
        logger.error("Test split not found at data/splits/test.parquet")
        sys.exit(1)

    test_df = pd.read_parquet(test_path)
    if args.nrows:
        test_df = test_df.head(args.nrows)

    logger.info(f"Loaded {len(test_df)} test samples for robustness benchmarking")

    vectorizers_path = models_dir / "tfidf_vectorizers.joblib"
    lr_path = models_dir / "logistic_regression.joblib"

    if not (vectorizers_path.exists() and lr_path.exists()):
        logger.error("Trained baseline model not found! Run train_baselines.py first.")
        sys.exit(1)

    vectorizers = joblib.load(vectorizers_path)
    model = joblib.load(lr_path)

    clean_texts = prepare_text(test_df)
    y_test = test_df["label"].values.astype(int)

    # 1. Clean Evaluation
    from scipy.sparse import hstack
    X_clean = hstack([vectorizers["word"].transform(clean_texts), vectorizers["char"].transform(clean_texts)])
    clean_probs = model.predict_proba(X_clean)[:, 1]
    clean_preds = model.predict(X_clean)
    clean_metrics = compute_all_metrics(y_test, clean_preds, clean_probs, prefix="clean")
    logger.info(f"Clean Performance: ROC-AUC={clean_metrics['clean_roc_auc']:.4f}, F1={clean_metrics['clean_f1']:.4f}")

    # 2. Perturbation Evaluations
    perturb_types = ["homoglyph", "zero_width", "typo", "url", "combined"]
    robustness_results = {
        "clean": {
            "roc_auc": clean_metrics["clean_roc_auc"],
            "f1": clean_metrics["clean_f1"],
            "recall": clean_metrics["clean_recall"],
        }
    }

    for p_type in perturb_types:
        logger.info(f"Applying perturbation: {p_type} (severity={args.severity})...")
        perturbed_texts = [apply_perturbations(t, p_type, severity=args.severity, seed=args.seed) for t in clean_texts]

        X_perturbed = hstack([vectorizers["word"].transform(perturbed_texts), vectorizers["char"].transform(perturbed_texts)])
        p_probs = model.predict_proba(X_perturbed)[:, 1]
        p_preds = model.predict(X_perturbed)

        m = compute_all_metrics(y_test, p_preds, p_probs, prefix=p_type)
        auc = m[f"{p_type}_roc_auc"]
        f1 = m[f"{p_type}_f1"]
        recall = m[f"{p_type}_recall"]

        auc_drop = round(float(clean_metrics["clean_roc_auc"] - auc), 4)
        f1_drop = round(float(clean_metrics["clean_f1"] - f1), 4)

        logger.info(f"[{p_type}] AUC={auc:.4f} (drop: {auc_drop:.4f}), F1={f1:.4f} (drop: {f1_drop:.4f})")
        robustness_results[p_type] = {
            "roc_auc": round(float(auc), 4),
            "f1": round(float(f1), 4),
            "recall": round(float(recall), 4),
            "auc_drop": auc_drop,
            "f1_drop": f1_drop,
        }

    # Save results
    results_path = output_dir / "robustness_benchmark_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(robustness_results, f, indent=2)

    # Markdown Table
    md = [
        "# Adversarial Perturbation Robustness Benchmark",
        "",
        "| Perturbation Type | Test ROC-AUC | AUC Drop (Δ) | Test F1 | F1 Drop (Δ) | Recall |",
        "|---|---|---|---|---|---|",
        f"| **Clean (Unperturbed)** | **{clean_metrics['clean_roc_auc']:.4f}** | — | **{clean_metrics['clean_f1']:.4f}** | — | **{clean_metrics['clean_recall']:.4f}** |",
    ]
    for p_type in perturb_types:
        r = robustness_results[p_type]
        md.append(f"| {p_type.capitalize()} | {r['roc_auc']:.4f} | -{r['auc_drop']:.4f} | {r['f1']:.4f} | -{r['f1_drop']:.4f} | {r['recall']:.4f} |")

    with open(output_dir / "robustness_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info(f"Robustness report saved to {output_dir / 'robustness_report.md'}")


if __name__ == "__main__":
    main()
