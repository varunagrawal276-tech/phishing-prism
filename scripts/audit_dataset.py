#!/usr/bin/env python3
"""
Dataset Audit Script — Stage 1

Inspects the Figshare phishing email datasets and produces:
- Schema report (columns, dtypes, row counts)
- Class distribution analysis
- Missingness analysis
- Text length statistics
- URL prevalence statistics
- Per-source summary
- Machine-readable JSON + CSV reports
- Markdown report
- Diagnostic plots

Usage:
    python scripts/audit_dataset.py [--download] [--nrows N]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import (
    download_all_datasets,
    get_dataset_summary,
    get_project_root,
    load_all_datasets,
    load_config,
)
from src.data.harmonization import discover_schema
from src.data.schemas import DatasetSchema
from src.training.seeds import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("audit_dataset")


def compute_text_stats(series: pd.Series, name: str) -> dict:
    """Compute text length statistics for a column."""
    lengths = series.dropna().astype(str).str.len()
    word_counts = series.dropna().astype(str).str.split().str.len()

    return {
        f"{name}_present": int(series.notna().sum()),
        f"{name}_missing": int(series.isna().sum()),
        f"{name}_empty": int((series.fillna("").astype(str).str.strip() == "").sum()),
        f"{name}_char_mean": round(float(lengths.mean()), 1) if len(lengths) > 0 else None,
        f"{name}_char_median": round(float(lengths.median()), 1) if len(lengths) > 0 else None,
        f"{name}_char_min": int(lengths.min()) if len(lengths) > 0 else None,
        f"{name}_char_max": int(lengths.max()) if len(lengths) > 0 else None,
        f"{name}_char_std": round(float(lengths.std()), 1) if len(lengths) > 0 else None,
        f"{name}_word_mean": round(float(word_counts.mean()), 1) if len(word_counts) > 0 else None,
        f"{name}_word_median": round(float(word_counts.median()), 1) if len(word_counts) > 0 else None,
        f"{name}_word_max": int(word_counts.max()) if len(word_counts) > 0 else None,
    }


def compute_url_stats(series: pd.Series) -> dict:
    """Compute URL prevalence statistics from text."""
    import re
    url_pattern = re.compile(
        r'https?://[^\s<>"\']+|www\.[^\s<>"\']+',
        re.IGNORECASE,
    )

    texts = series.dropna().astype(str)
    url_counts = texts.apply(lambda x: len(url_pattern.findall(x)))

    return {
        "emails_with_urls": int((url_counts > 0).sum()),
        "emails_with_urls_pct": round(float((url_counts > 0).mean() * 100), 2),
        "url_count_mean": round(float(url_counts.mean()), 2),
        "url_count_max": int(url_counts.max()) if len(url_counts) > 0 else 0,
        "total_urls_found": int(url_counts.sum()),
    }


def audit_single_dataset(
    df: pd.DataFrame,
    source_name: str,
    schema: DatasetSchema,
) -> dict:
    """Perform comprehensive audit of a single dataset."""
    audit = {
        "source": source_name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    # Schema discovery results
    audit["schema"] = {
        "label_column": schema.label_column,
        "label_values": [str(v) for v in (schema.label_values or [])],
        "subject_column": schema.subject_column,
        "body_column": schema.body_column,
        "sender_column": schema.sender_column,
        "receiver_column": schema.receiver_column,
        "date_column": schema.date_column,
        "url_column": schema.url_column,
    }

    # Missing value analysis
    audit["missing"] = {
        col: {
            "count": int(df[col].isnull().sum()),
            "pct": round(float(df[col].isnull().mean() * 100), 2),
        }
        for col in df.columns
    }

    # Class distribution
    if schema.label_column and schema.label_column in df.columns:
        label_counts = df[schema.label_column].value_counts()
        audit["class_distribution"] = {
            str(k): int(v) for k, v in label_counts.items()
        }
        audit["class_percentages"] = {
            str(k): round(float(v / len(df) * 100), 2)
            for k, v in label_counts.items()
        }

    # Text statistics
    if schema.body_column and schema.body_column in df.columns:
        audit["body_stats"] = compute_text_stats(df[schema.body_column], "body")
        audit["url_stats"] = compute_url_stats(df[schema.body_column])

    if schema.subject_column and schema.subject_column in df.columns:
        audit["subject_stats"] = compute_text_stats(df[schema.subject_column], "subject")

    # Sender/receiver availability
    audit["metadata_availability"] = {
        "sender": schema.sender_column is not None,
        "receiver": schema.receiver_column is not None,
        "date": schema.date_column is not None,
    }

    return audit


def generate_plots(audits: list[dict], output_dir: Path) -> list[Path]:
    """Generate diagnostic plots from audit results."""
    plot_paths = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Row counts per dataset
    fig, ax = plt.subplots(figsize=(10, 6))
    sources = [a["source"] for a in audits]
    rows = [a["rows"] for a in audits]
    colors = plt.cm.Set2(np.linspace(0, 1, len(sources)))
    bars = ax.barh(sources, rows, color=colors, edgecolor="gray")
    ax.set_xlabel("Number of Emails")
    ax.set_title("Dataset Size by Source")
    ax.bar_label(bars, fmt="%d", padding=3)
    plt.tight_layout()
    path = output_dir / "dataset_sizes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plot_paths.append(path)

    # 2. Class distribution per dataset
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, audit in enumerate(audits):
        ax = axes[i]
        if "class_distribution" in audit:
            labels = list(audit["class_distribution"].keys())
            counts = list(audit["class_distribution"].values())
            ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title(audit["source"], fontsize=10)
    if len(audits) < 8:
        axes[-1].set_visible(False)
    fig.suptitle("Class Distribution by Source", fontsize=14)
    plt.tight_layout()
    path = output_dir / "class_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plot_paths.append(path)

    # 3. Body length distributions
    fig, ax = plt.subplots(figsize=(12, 6))
    for audit in audits:
        if "body_stats" in audit and audit["body_stats"].get("body_char_mean") is not None:
            mean_len = audit["body_stats"]["body_char_mean"]
            std_len = audit["body_stats"].get("body_char_std", 0) or 0
            ax.barh(
                audit["source"],
                mean_len,
                xerr=std_len,
                capsize=3,
                alpha=0.7,
            )
    ax.set_xlabel("Body Length (characters)")
    ax.set_title("Average Email Body Length by Source")
    plt.tight_layout()
    path = output_dir / "body_lengths.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plot_paths.append(path)

    # 4. URL prevalence
    fig, ax = plt.subplots(figsize=(10, 6))
    url_pcts = []
    url_sources = []
    for audit in audits:
        if "url_stats" in audit:
            url_sources.append(audit["source"])
            url_pcts.append(audit["url_stats"]["emails_with_urls_pct"])
    if url_pcts:
        ax.barh(url_sources, url_pcts, color="coral", edgecolor="gray")
        ax.set_xlabel("% of Emails Containing URLs")
        ax.set_title("URL Prevalence by Source")
        ax.set_xlim(0, 100)
    plt.tight_layout()
    path = output_dir / "url_prevalence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plot_paths.append(path)

    logger.info(f"Generated {len(plot_paths)} plots in {output_dir}")
    return plot_paths


def generate_markdown_report(audits: list[dict], output_path: Path) -> None:
    """Generate a human-readable Markdown audit report."""
    lines = [
        "# Dataset Audit Report",
        "",
        f"**Generated by:** `scripts/audit_dataset.py`",
        f"**Number of sources:** {len(audits)}",
        f"**Total rows:** {sum(a['rows'] for a in audits):,}",
        "",
        "## Summary Table",
        "",
        "| Source | Rows | Columns | Label Col | Body Col | Subject Col | Sender | Receiver | Date |",
        "|--------|------|---------|-----------|----------|-------------|--------|----------|------|",
    ]

    for a in audits:
        s = a["schema"]
        lines.append(
            f"| {a['source']} | {a['rows']:,} | {a['columns']} | "
            f"{s['label_column'] or '❌'} | {s['body_column'] or '❌'} | "
            f"{s['subject_column'] or '❌'} | "
            f"{'✅' if s['sender_column'] else '❌'} | "
            f"{'✅' if s['receiver_column'] else '❌'} | "
            f"{'✅' if s['date_column'] else '❌'} |"
        )

    lines.extend([
        "",
        "## Class Distribution",
        "",
        "| Source | Classes | Distribution |",
        "|--------|---------|--------------|",
    ])

    for a in audits:
        if "class_distribution" in a:
            dist = ", ".join(f"{k}: {v:,}" for k, v in a["class_distribution"].items())
            pcts = ", ".join(f"{k}: {v}%" for k, v in a["class_percentages"].items())
            lines.append(f"| {a['source']} | {dist} | {pcts} |")

    lines.extend([
        "",
        "## Text Statistics",
        "",
        "| Source | Body Mean Len | Body Median Len | Body Max Len | Word Mean | Empty Bodies |",
        "|--------|---------------|-----------------|--------------|-----------|--------------|",
    ])

    for a in audits:
        bs = a.get("body_stats", {})
        lines.append(
            f"| {a['source']} | "
            f"{bs.get('body_char_mean', 'N/A')} | "
            f"{bs.get('body_char_median', 'N/A')} | "
            f"{bs.get('body_char_max', 'N/A')} | "
            f"{bs.get('body_word_mean', 'N/A')} | "
            f"{bs.get('body_empty', 'N/A')} |"
        )

    lines.extend([
        "",
        "## URL Prevalence",
        "",
        "| Source | Emails with URLs | URL % | Mean URL Count | Total URLs |",
        "|--------|-----------------|-------|----------------|------------|",
    ])

    for a in audits:
        us = a.get("url_stats", {})
        lines.append(
            f"| {a['source']} | "
            f"{us.get('emails_with_urls', 'N/A')} | "
            f"{us.get('emails_with_urls_pct', 'N/A')}% | "
            f"{us.get('url_count_mean', 'N/A')} | "
            f"{us.get('total_urls_found', 'N/A')} |"
        )

    lines.extend([
        "",
        "## Per-Source Column Details",
        "",
    ])

    for a in audits:
        lines.append(f"### {a['source']}")
        lines.append(f"- **Rows:** {a['rows']:,}")
        lines.append(f"- **Columns:** {a['columns']}")
        lines.append(f"- **Column names:** `{a['column_names']}`")
        lines.append(f"- **Dtypes:** `{a['dtypes']}`")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Markdown report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit phishing email datasets")
    parser.add_argument("--download", action="store_true", help="Download datasets from Figshare")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config("data.yaml")
    project_root = get_project_root()

    # Output directories
    audit_dir = project_root / "artifacts" / "dataset_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download if requested
    if args.download:
        logger.info("=== Downloading datasets from Figshare ===")
        download_all_datasets(config=config)

    # Step 2: Load datasets
    logger.info("=== Loading datasets ===")
    datasets = load_all_datasets(config=config, nrows=args.nrows)

    if not datasets:
        logger.error(
            "No datasets found! Place CSV files in data/raw/ or use --download flag."
        )
        sys.exit(1)

    # Step 3: Generate summary
    logger.info("=== Dataset summary ===")
    summary_df = get_dataset_summary(datasets)
    print("\n" + summary_df.to_string(index=False))
    summary_df.to_csv(audit_dir / "dataset_summary.csv", index=False)

    # Step 4: Audit each dataset
    logger.info("=== Auditing individual datasets ===")
    audits = []
    schemas = {}

    for source_name, df in datasets.items():
        logger.info(f"\n--- {source_name} ---")
        schema = discover_schema(df, source_name)
        schemas[source_name] = schema
        audit = audit_single_dataset(df, source_name, schema)
        audits.append(audit)

        # Print key findings
        logger.info(f"  Rows: {audit['rows']:,}")
        logger.info(f"  Columns: {audit['columns']}")
        logger.info(f"  Label column: {schema.label_column}")
        logger.info(f"  Label values: {schema.label_values}")
        if "class_distribution" in audit:
            logger.info(f"  Class distribution: {audit['class_distribution']}")
        logger.info(f"  Body column: {schema.body_column}")
        logger.info(f"  Subject column: {schema.subject_column}")

    # Step 5: Save JSON report
    json_path = audit_dir / "audit_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audits, f, indent=2, default=str)
    logger.info(f"JSON report saved to {json_path}")

    # Step 6: Generate Markdown report
    md_path = audit_dir / "audit_report.md"
    generate_markdown_report(audits, md_path)

    # Step 7: Generate plots
    logger.info("=== Generating plots ===")
    generate_plots(audits, audit_dir)

    # Step 8: Summary
    total_rows = sum(a["rows"] for a in audits)
    logger.info(f"\n{'='*60}")
    logger.info(f"AUDIT COMPLETE")
    logger.info(f"  Sources audited: {len(audits)}")
    logger.info(f"  Total rows: {total_rows:,}")
    logger.info(f"  Artifacts saved to: {audit_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
