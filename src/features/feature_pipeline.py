"""
Feature pipeline for PRISM-Phish.

Orchestrates all feature extraction into a single feature matrix.
Features are fitted on training data only and saved for inference reuse.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.lexical import compute_lexical_features
from src.features.structural import compute_structural_features
from src.features.url_features import compute_url_features, extract_urls
from src.features.header_features import compute_header_features
from src.preprocessing.html import extract_html_features

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Unified feature extraction pipeline.

    Extracts all deterministic features and applies normalization.
    Scaler is fitted on training data only.
    """

    def __init__(self, normalization: str = "standard"):
        self.normalization = normalization
        self.scaler = None
        self.feature_names: list[str] = []
        self._fitted = False

    def extract_features_single(self, row: pd.Series) -> dict:
        """Extract all features from a single email row."""
        features = {}

        body = str(row.get("body", "")) if pd.notna(row.get("body")) else ""
        subject = str(row.get("subject", "")) if pd.notna(row.get("subject")) else ""
        sender = str(row.get("sender", "")) if pd.notna(row.get("sender")) else ""
        receiver = str(row.get("receiver", "")) if pd.notna(row.get("receiver")) else ""

        # Lexical features
        features.update(compute_lexical_features(body, subject))

        # Structural features
        features.update(compute_structural_features(body, subject))

        # URL features
        urls = extract_urls(body)
        features.update(compute_url_features(urls))

        # Header features
        features.update(compute_header_features(sender, receiver))

        # HTML features
        features.update(extract_html_features(body))

        return features

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for all rows in a DataFrame."""
        logger.info(f"Extracting features for {len(df)} samples...")

        feature_dicts = []
        records = df.to_dict("records")
        for row in records:
            feature_dicts.append(self.extract_features_single(row))

        features_df = pd.DataFrame(feature_dicts)

        # Convert booleans to int
        bool_cols = features_df.select_dtypes(include="bool").columns
        features_df[bool_cols] = features_df[bool_cols].astype(int)

        self.feature_names = list(features_df.columns)
        logger.info(f"Extracted {len(self.feature_names)} features")

        return features_df

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit scaler on training data and transform."""
        features_df = self.extract_features(df)
        values = features_df.values.astype(np.float32)

        # Handle NaN/inf
        values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.normalization == "standard":
            self.scaler = StandardScaler()
        elif self.normalization == "robust":
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        elif self.normalization == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        if self.scaler:
            values = self.scaler.fit_transform(values)

        self._fitted = True
        return values

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform using fitted scaler (for val/test data)."""
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform first.")

        features_df = self.extract_features(df)
        values = features_df.values.astype(np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.scaler:
            values = self.scaler.transform(values)

        return values

    def save(self, path: Path) -> None:
        """Save pipeline state (scaler + feature names) for inference."""
        import pickle
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open(path / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f)

        logger.info(f"Feature pipeline saved to {path}")

    def load(self, path: Path) -> None:
        """Load pipeline state from disk."""
        import pickle

        with open(path / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(path / "feature_names.json", "r") as f:
            self.feature_names = json.load(f)

        self._fitted = True
        logger.info(f"Feature pipeline loaded from {path}")
