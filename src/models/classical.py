"""
Classical ML models for PRISM-Phish baselines.

Implements TF-IDF + Logistic Regression, Linear SVM, and XGBoost/Extra Trees.
Uses word + character n-grams as specified.
"""

import logging
import time
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.sparse import hstack

logger = logging.getLogger(__name__)


def build_tfidf_features(
    train_texts: list[str],
    val_texts: Optional[list[str]] = None,
    test_texts: Optional[list[str]] = None,
    max_features_word: int = 50000,
    max_features_char: int = 50000,
    word_ngram_range: tuple = (1, 2),
    char_ngram_range: tuple = (3, 5),
    sublinear_tf: bool = True,
    min_df: int = 2,
) -> tuple:
    """
    Build TF-IDF features combining word and character n-grams.

    Returns fitted vectorizers and transformed features.
    """
    logger.info("Building TF-IDF features (word + char n-grams)")

    # Word n-gram vectorizer
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        max_features=max_features_word,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        dtype=np.float32,
    )

    # Character n-gram vectorizer
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        max_features=max_features_char,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        dtype=np.float32,
    )

    # Fit on training data only
    X_train_word = word_vec.fit_transform(train_texts)
    X_train_char = char_vec.fit_transform(train_texts)
    X_train = hstack([X_train_word, X_train_char])

    logger.info(
        f"  Word features: {X_train_word.shape[1]}, "
        f"Char features: {X_train_char.shape[1]}, "
        f"Total: {X_train.shape[1]}"
    )

    results = {"train": X_train}
    vectorizers = {"word": word_vec, "char": char_vec}

    if val_texts is not None:
        X_val = hstack([word_vec.transform(val_texts), char_vec.transform(val_texts)])
        results["val"] = X_val

    if test_texts is not None:
        X_test = hstack([word_vec.transform(test_texts), char_vec.transform(test_texts)])
        results["test"] = X_test

    return results, vectorizers


def train_logistic_regression(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    C_values: list[float] = None,
    max_iter: int = 1000,
) -> dict:
    """Train Logistic Regression with hyperparameter search on validation."""
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0]

    logger.info(f"Training Logistic Regression (C={C_values})")
    start = time.time()

    best_model = None
    best_score = -1
    best_c = None

    for c in C_values:
        model = LogisticRegression(C=c, max_iter=max_iter, solver="lbfgs", random_state=42)
        model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            score = model.score(X_val, y_val)
            logger.info(f"  C={c}: val_accuracy={score:.4f}")
            if score > best_score:
                best_score = score
                best_model = model
                best_c = c
        else:
            best_model = model
            best_c = c

    duration = time.time() - start
    logger.info(f"  Best C={best_c}, trained in {duration:.1f}s")

    return {
        "model": best_model,
        "best_c": best_c,
        "best_val_score": best_score,
        "training_time": duration,
    }


def train_linear_svm(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    C_values: list[float] = None,
    max_iter: int = 5000,
) -> dict:
    """Train Linear SVM with calibration for probability output."""
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0]

    logger.info(f"Training Linear SVM (C={C_values})")
    start = time.time()

    best_model = None
    best_score = -1
    best_c = None

    for c in C_values:
        svm = LinearSVC(C=c, max_iter=max_iter, random_state=42)
        # Wrap with CalibratedClassifierCV for probability estimates
        model = CalibratedClassifierCV(svm, cv=3)
        model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            score = model.score(X_val, y_val)
            logger.info(f"  C={c}: val_accuracy={score:.4f}")
            if score > best_score:
                best_score = score
                best_model = model
                best_c = c
        else:
            best_model = model
            best_c = c

    duration = time.time() - start
    logger.info(f"  Best C={best_c}, trained in {duration:.1f}s")

    return {
        "model": best_model,
        "best_c": best_c,
        "best_val_score": best_score,
        "training_time": duration,
    }


def train_xgboost(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> dict:
    """Train XGBoost classifier."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("XGBoost not available, skipping")
        return None

    logger.info(f"Training XGBoost (n_estimators={n_estimators}, max_depth={max_depth})")
    start = time.time()

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False,
    )

    duration = time.time() - start
    val_score = model.score(X_val, y_val) if X_val is not None else None
    logger.info(f"  Trained in {duration:.1f}s, val_accuracy={val_score}")

    return {
        "model": model,
        "training_time": duration,
        "val_score": val_score,
    }
