#!/usr/bin/env python3
"""
Build Processed Dataset Script — Stage 2

Pipeline steps:
1. Load raw datasets from data/raw/
2. Harmonize schemas & normalize labels -> data/processed/harmonized.parquet
3. Run exact and near-duplicate detection -> data/processed/deduplicated.parquet
4. Generate deduplication statistics report

Usage:
    python scripts/build_processed_dataset.py [--threshold 0.85] [--nrows N]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.deduplication import compute_duplicate_hashes, find_exact_duplicates
from src.data.harmonization import harmonize_all_datasets
from src.data.loaders import load_all_datasets, load_config
from src.data.near_duplicate import find_near_duplicates
from src.training.seeds import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_processed_dataset")


def main():
    parser = argparse.ArgumentParser(description="Harmonize and deduplicate raw email datasets")
    parser.add_argument("--threshold", type=float, default=0.85, help="Jaccard similarity threshold for near-duplicates")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows per dataset for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config("data.yaml")
    processed_dir = PROJECT_ROOT / config["paths"]["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw datasets
    logger.info("=== Loading raw datasets ===")
    raw_datasets = load_all_datasets(config=config, nrows=args.nrows)
    if not raw_datasets:
        logger.error("No raw datasets found in data/raw/!")
        sys.exit(1)

    # 2. Harmonize
    logger.info("=== Harmonizing schemas and labels ===")
    harmonized_df, schemas, label_mappings = harmonize_all_datasets(raw_datasets)
    
    harmonized_path = processed_dir / "harmonized.parquet"
    harmonized_df.to_parquet(harmonized_path, index=False)
    logger.info(f"Saved harmonized dataset ({len(harmonized_df)} rows) to {harmonized_path}")

    # 3. Exact deduplication
    logger.info("=== Computing exact duplicate hashes ===")
    df_hashed = compute_duplicate_hashes(harmonized_df)
    df_dedup = find_exact_duplicates(df_hashed, hash_column="subj_body_hash")

    # 4. Near-duplicate detection
    logger.info(f"=== Computing MinHash LSH near-duplicates (threshold={args.threshold}) ===")
    df_dedup, near_dup_stats = find_near_duplicates(
        df_dedup,
        text_column="body",
        threshold=args.threshold,
    )

    # Combine exact and near-duplicate cluster IDs into unified_cluster_id
    df_dedup["unified_cluster_id"] = df_dedup["near_dup_cluster_id"].fillna(df_dedup["dedup_cluster_id"])

    dedup_path = processed_dir / "deduplicated.parquet"
    df_dedup.to_parquet(dedup_path, index=False)
    logger.info(f"Saved deduplicated dataset ({len(df_dedup)} rows) to {dedup_path}")

    # Save summary stats
    stats = {
        "total_harmonized_rows": len(harmonized_df),
        "total_deduplicated_rows": len(df_dedup),
        "exact_duplicate_clusters": int(df_dedup["dedup_cluster_id"].nunique()),
        "near_duplicate_stats": near_dup_stats,
        "sources": {str(k): int(v) for k, v in harmonized_df["dataset_id"].value_counts().items()},
        "label_distribution": {str(k): int(v) for k, v in harmonized_df["label"].value_counts().items()},
    }
    stats_path = processed_dir / "deduplication_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Saved deduplication statistics to {stats_path}")


if __name__ == "__main__":
    main()
