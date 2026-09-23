"""
Error Diagnostics & Failure Mode Analysis for PRISM-Phish.

Systematically identifies, categorizes, and audits False Positives (FP)
and False Negatives (FN) across email corpora.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Analyzes error distributions across metadata attributes.
    """

    def __init__(self, df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
        self.df = df.copy()
        self.df["y_true"] = y_true
        self.df["y_pred"] = y_pred
        self.df["y_prob"] = y_prob

        # Classify outcomes
        self.df["error_type"] = "CORRECT"
        self.df.loc[(self.df["y_true"] == 0) & (self.df["y_pred"] == 1), "error_type"] = "FALSE_POSITIVE"
        self.df.loc[(self.df["y_true"] == 1) & (self.df["y_pred"] == 0), "error_type"] = "FALSE_NEGATIVE"

    def compute_diagnostics(self) -> dict:
        """Compute holistic diagnostic breakdown."""
        total = len(self.df)
        fp_mask = self.df["error_type"] == "FALSE_POSITIVE"
        fn_mask = self.df["error_type"] == "FALSE_NEGATIVE"

        num_fp = int(fp_mask.sum())
        num_fn = int(fn_mask.sum())
        total_errors = num_fp + num_fn

        diagnostics = {
            "total_samples": total,
            "total_errors": total_errors,
            "error_rate": float(total_errors / total) if total > 0 else 0.0,
            "false_positives": {
                "count": num_fp,
                "fp_rate": float(num_fp / max((self.df["y_true"] == 0).sum(), 1)),
            },
            "false_negatives": {
                "count": num_fn,
                "fn_rate": float(num_fn / max((self.df["y_true"] == 1).sum(), 1)),
            },
        }

        # Breakdown by dataset_id / source if available
        if "dataset_id" in self.df.columns:
            source_breakdown = {}
            for source, grp in self.df.groupby("dataset_id"):
                s_fp = int((grp["error_type"] == "FALSE_POSITIVE").sum())
                s_fn = int((grp["error_type"] == "FALSE_NEGATIVE").sum())
                source_breakdown[str(source)] = {
                    "total": len(grp),
                    "fp": s_fp,
                    "fn": s_fn,
                    "error_rate": float((s_fp + s_fn) / len(grp)),
                }
            diagnostics["by_source"] = source_breakdown

        # Sample worst false positives (highest predicted prob of phishing)
        fps = self.df[fp_mask].sort_values("y_prob", ascending=False)
        diagnostics["worst_false_positives"] = [
            {
                "subject": str(row.get("subject", ""))[:80],
                "pred_prob": float(row["y_prob"]),
                "source": str(row.get("dataset_id", "unknown")),
            }
            for _, row in fps.head(5).iterrows()
        ]

        # Sample worst false negatives (lowest predicted prob of phishing)
        fns = self.df[fn_mask].sort_values("y_prob", ascending=True)
        diagnostics["worst_false_negatives"] = [
            {
                "subject": str(row.get("subject", ""))[:80],
                "pred_prob": float(row["y_prob"]),
                "source": str(row.get("dataset_id", "unknown")),
            }
            for _, row in fns.head(5).iterrows()
        ]

        return diagnostics

    def save_report(self, diagnostics: dict, output_path: Union[str, Path]):
        """Save report to JSON and Markdown."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        json_path = out.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)

        md_path = out.with_suffix(".md")
        lines = [
            "# Error Diagnostics & Failure Mode Report",
            "",
            f"- **Total Evaluated Samples:** {diagnostics['total_samples']:,}",
            f"- **Total Errors:** {diagnostics['total_errors']:,} ({diagnostics['error_rate']*100:.2f}%)",
            f"- **False Positives (FP):** {diagnostics['false_positives']['count']:,} (FPR: {diagnostics['false_positives']['fp_rate']*100:.2f}%)",
            f"- **False Negatives (FN):** {diagnostics['false_negatives']['count']:,} (FNR: {diagnostics['false_negatives']['fn_rate']*100:.2f}%)",
            "",
        ]

        if "by_source" in diagnostics:
            lines.extend([
                "## Error Distribution by Source",
                "",
                "| Dataset Source | Samples | FP | FN | Error Rate |",
                "|---|---|---|---|---|",
            ])
            for src, stat in diagnostics["by_source"].items():
                lines.append(f"| {src} | {stat['total']:,} | {stat['fp']} | {stat['fn']} | {stat['error_rate']*100:.2f}% |")
            lines.append("")

        if diagnostics.get("worst_false_positives"):
            lines.extend([
                "## Top False Positives (Legitimate flagged as Phishing)",
                "",
                "| Source | Predicted Phish Prob | Subject Preview |",
                "|---|---|---|",
            ])
            for fp in diagnostics["worst_false_positives"]:
                lines.append(f"| {fp['source']} | {fp['pred_prob']:.4f} | {fp['subject']} |")
            lines.append("")

        if diagnostics.get("worst_false_negatives"):
            lines.extend([
                "## Top False Negatives (Phishing missed by Detector)",
                "",
                "| Source | Predicted Phish Prob | Subject Preview |",
                "|---|---|---|",
            ])
            for fn in diagnostics["worst_false_negatives"]:
                lines.append(f"| {fn['source']} | {fn['pred_prob']:.4f} | {fn['subject']} |")
            lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Saved error diagnostics to {json_path} and {md_path}")
