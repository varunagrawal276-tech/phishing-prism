#!/usr/bin/env python3
"""
Run Explainability and Error Diagnostics — Stage 10

Generates:
1. SHAP global feature attributions (artifacts/explainability/shap_summary.md)
2. Error failure mode diagnostics (artifacts/explainability/error_analysis.md)
"""

import json
import logging
import sys
from pathlib import Path
from scipy.sparse import hstack
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.error_analysis import ErrorAnalyzer
from src.explainability.shap_explainer import TabularSHAPExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_explainability")


def prepare_text(df: pd.DataFrame) -> list[str]:
    subj = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    return (subj + " " + body).tolist()


def main():
    output_dir = PROJECT_ROOT / "artifacts" / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load test data and trained baseline
    logger.info("Loading test data and baseline artifacts...")
    test_df = pd.read_parquet(PROJECT_ROOT / "data" / "splits" / "test.parquet")
    model = joblib.load(PROJECT_ROOT / "artifacts" / "baselines" / "logistic_regression.joblib")
    vectorizers = joblib.load(PROJECT_ROOT / "artifacts" / "baselines" / "tfidf_vectorizers.joblib")

    test_texts = prepare_text(test_df)
    y_test = test_df["label"].values.astype(int)

    logger.info("Vectorizing test samples...")
    X_word = vectorizers["word"].transform(test_texts)
    X_char = vectorizers["char"].transform(test_texts)
    X_test = hstack([X_word, X_char])

    # 2. Run Predictions
    logger.info("Generating predictions on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 3. Error Analysis
    logger.info("Running Error Diagnostics...")
    analyzer = ErrorAnalyzer(test_df, y_test, y_pred, y_prob)
    diagnostics = analyzer.compute_diagnostics()
    analyzer.save_report(diagnostics, output_dir / "error_analysis")

    # 4. Feature Attribution
    logger.info("Extracting top predictive features from baseline model...")
    word_feature_names = [f"word:{w}" for w in vectorizers["word"].get_feature_names_out()]
    char_feature_names = [f"char:{c}" for c in vectorizers["char"].get_feature_names_out()]
    all_features = word_feature_names + char_feature_names

    explainer = TabularSHAPExplainer(model, all_features)
    explanation = explainer.explain_dataset(X_test, max_samples=1000)
    explainer.save_report(explanation, output_dir / "shap_summary")

    logger.info("Explainability stage complete! All reports saved in artifacts/explainability/")


if __name__ == "__main__":
    main()
