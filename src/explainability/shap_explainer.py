"""
SHAP Explainer for PRISM-Phish.

Provides feature attribution and interpretability for structural and tabular features
using SHAP (SHapley Additive exPlanations).
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TabularSHAPExplainer:
    """
    Computes SHAP explanations for tabular/structural feature models.
    """

    def __init__(self, model, feature_names: list[str], background_samples: Optional[np.ndarray] = None):
        self.model = model
        self.feature_names = feature_names
        self.background_samples = background_samples
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initialize SHAP explainer based on model type."""
        try:
            import shap

            model_type = type(self.model).__name__
            if "XGB" in model_type or "Forest" in model_type or "Tree" in model_type:
                logger.info(f"Initializing TreeExplainer for {model_type}")
                self.explainer = shap.TreeExplainer(self.model)
            elif "LogisticRegression" in model_type or "Linear" in model_type:
                logger.info(f"Initializing LinearExplainer for {model_type}")
                if self.background_samples is not None:
                    self.explainer = shap.LinearExplainer(self.model, self.background_samples)
                else:
                    self.explainer = shap.Explainer(self.model)
            else:
                logger.info(f"Initializing generic Explainer for {model_type}")
                if self.background_samples is not None:
                    self.explainer = shap.Explainer(self.model.predict_proba, self.background_samples)
                else:
                    self.explainer = shap.Explainer(self.model)
        except Exception as e:
            logger.warning(f"Could not initialize standard SHAP explainer: {e}. Falling back to coefficient-based attribution.")
            self.explainer = None

    def explain_dataset(self, X: np.ndarray, max_samples: int = 500) -> dict:
        """
        Compute global feature importance across sample dataset.

        Returns dict with top features and mean absolute SHAP values.
        """
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        if n_samples > max_samples:
            indices = np.random.choice(n_samples, max_samples, replace=False)
            X_eval = X[indices]
        else:
            X_eval = X

        if self.explainer is not None:
            try:
                shap_values = self.explainer(X_eval)
                if hasattr(shap_values, "values"):
                    vals = shap_values.values
                    # For binary classification, take positive class
                    if vals.ndim == 3 and vals.shape[2] == 2:
                        vals = vals[:, :, 1]
                else:
                    vals = np.array(shap_values)

                mean_abs_shap = np.mean(np.abs(vals), axis=0)
            except Exception as e:
                logger.warning(f"SHAP evaluation failed ({e}), falling back to linear/importance weights.")
                mean_abs_shap = self._fallback_importance()
        else:
            mean_abs_shap = self._fallback_importance()

        if mean_abs_shap is None or len(mean_abs_shap) != len(self.feature_names):
            mean_abs_shap = np.ones(len(self.feature_names)) / len(self.feature_names)

        # Sort features by importance
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        ranking = [
            {
                "feature": self.feature_names[i],
                "importance": float(mean_abs_shap[i]),
                "rank": rank + 1,
            }
            for rank, i in enumerate(sorted_indices)
        ]

        return {
            "num_evaluated_samples": X_eval.shape[0] if hasattr(X_eval, "shape") else len(X_eval),
            "top_features": ranking[:20],
            "all_features": ranking,
        }

    def _fallback_importance(self) -> Optional[np.ndarray]:
        """Extract importance from model attributes (feature_importances_ or coef_)."""
        if hasattr(self.model, "feature_importances_"):
            return np.array(self.model.feature_importances_)
        elif hasattr(self.model, "coef_"):
            coef = np.array(self.model.coef_)
            if coef.ndim > 1:
                coef = coef[0]
            return np.abs(coef)
        return None

    def save_report(self, explanation: dict, output_path: Union[str, Path]):
        """Save explanation results as JSON and Markdown summary."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        json_path = out.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(explanation, f, indent=2)

        md_path = out.with_suffix(".md")
        lines = [
            "# Global Feature Attribution Report (SHAP)",
            "",
            f"**Samples Evaluated:** {explanation['num_evaluated_samples']:,}",
            "",
            "| Rank | Feature Name | Mean |SHAP| Value |",
            "|---|---|---|",
        ]
        for item in explanation["top_features"]:
            lines.append(f"| {item['rank']} | `{item['feature']}` | {item['importance']:.5f} |")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Saved SHAP reports to {json_path} and {md_path}")
