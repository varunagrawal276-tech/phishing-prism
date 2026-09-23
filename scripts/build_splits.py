#!/usr/bin/env python3
"""
Build Dataset Splits Script — Stage 2

Creates:
1. In-domain IID splits (train.parquet, val.parquet, test.parquet) with group-awareness
2. Leave-One-Source-Out (LOSO) cross-dataset splits in data/splits/loso/

Usage:
    python scripts/build_splits.py [--seed 42]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_config
from src.data.splitting import create_iid_split, create_loso_splits
from src.training.seeds import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_splits")


def main():
    parser = argparse.ArgumentParser(description="Build IID and LOSO splits for model training and evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test set fraction")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation set fraction")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config("data.yaml")
    
    processed_file = PROJECT_ROOT / config["paths"]["processed"] / "deduplicated.parquet"
    if not processed_file.exists():
        processed_file = PROJECT_ROOT / config["paths"]["processed"] / "harmonized.parquet"

    if not processed_file.exists():
        logger.error("No processed dataset found! Run scripts/build_processed_dataset.py first.")
        sys.exit(1)

    logger.info(f"Loading processed dataset from {processed_file}")
    df = pd.read_parquet(processed_file)

    splits_dir = PROJECT_ROOT / config["paths"]["splits"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create IID Splits
    logger.info("=== Creating IID train/val/test splits ===")
    train_df, val_df, test_df = create_iid_split(
        df,
        test_size=args.test_size,
        val_size=args.val_size,
        stratify_col="label",
        group_col="unified_cluster_id" if "unified_cluster_id" in df.columns else None,
        seed=args.seed,
    )

    train_df.to_parquet(splits_dir / "train.parquet", index=False)
    val_df.to_parquet(splits_dir / "val.parquet", index=False)
    test_df.to_parquet(splits_dir / "test.parquet", index=False)
    logger.info(f"Saved IID splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Create LOSO Splits
    logger.info("=== Creating Leave-One-Source-Out (LOSO) splits ===")
    loso_dir = splits_dir / "loso"
    loso_dir.mkdir(parents=True, exist_ok=True)
    loso_splits = create_loso_splits(df, val_size=args.val_size, seed=args.seed)

    loso_summary = {}
    for held_out_source, split_dict in loso_splits.items():
        source_dir = loso_dir / held_out_source
        source_dir.mkdir(parents=True, exist_ok=True)
        split_dict["train"].to_parquet(source_dir / "train.parquet", index=False)
        split_dict["val"].to_parquet(source_dir / "val.parquet", index=False)
        split_dict["test"].to_parquet(source_dir / "test.parquet", index=False)
        loso_summary[held_out_source] = {
            "train_rows": len(split_dict["train"]),
            "val_rows": len(split_dict["val"]),
            "test_rows": len(split_dict["test"]),
            "train_sources": [str(s) for s in split_dict["train"]["dataset_id"].unique()],
            "held_out_source": str(held_out_source),
        }

    with open(loso_dir / "loso_summary.json", "w", encoding="utf-8") as f:
        json.dump(loso_summary, f, indent=2, default=str)
    logger.info(f"Saved LOSO splits for {len(loso_splits)} sources in {loso_dir}")


if __name__ == "__main__":
    main()
