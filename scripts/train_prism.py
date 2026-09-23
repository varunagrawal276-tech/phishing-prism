#!/usr/bin/env python3
"""
Train PRISM-Phish Model — Stage 5

Trains the unified PRISM-Phish model:
1. DeBERTa-v3 Text Encoder
2. Structural / URL / Header Feature MLP
3. Gated Cross-Attention Fusion
4. Phishing Classifier Head
5. Domain-Adversarial Gradient Reversal Layer (GRL)
6. Perturbation Consistency Regularization

Usage:
    python scripts/train_prism.py [--epochs 3] [--batch-size 16] [--grl] [--consistency]
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_config
from src.evaluation.metrics import compute_all_metrics
from src.features.feature_pipeline import FeaturePipeline
from src.models.consistency import ConsistencyLoss
from src.models.prism_phish import PRISMPhish
from src.training.losses import CombinedLoss
from src.training.seeds import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_prism")


class PRISMDataset(Dataset):
    """Dataset for multimodal text + structural features + source IDs."""

    def __init__(self, texts: list[str], struct_features: np.ndarray, labels: list[int], source_ids: list[int], tokenizer, max_length: int = 512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.struct_features = torch.tensor(struct_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.source_ids = torch.tensor(source_ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["struct_features"] = self.struct_features[idx]
        item["labels"] = self.labels[idx]
        item["source_ids"] = self.source_ids[idx]
        return item


def prism_collate_fn(batch, tokenizer):
    """Pad input_ids and stack tensors."""
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    struct_features = torch.stack([item["struct_features"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    source_ids = torch.stack([item["source_ids"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "struct_features": struct_features,
        "labels": labels,
        "source_ids": source_ids,
    }


def prepare_text(df: pd.DataFrame) -> list[str]:
    subj = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    return (subj + " " + body).tolist()


def evaluate_prism(model, dataloader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            struct_features = batch["struct_features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, structural_features=struct_features)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return compute_all_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs), prefix="test")


def main():
    parser = argparse.ArgumentParser(description="Train PRISM-Phish model")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-small", help="Text encoder backbone")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max token sequence length")
    parser.add_argument("--grl", action="store_true", default=True, help="Enable domain adversarial GRL")
    parser.add_argument("--grl-lambda", type=float, default=0.2, help="Domain loss weight")
    parser.add_argument("--consistency", action="store_true", default=True, help="Enable consistency loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows for rapid testing")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    splits_dir = PROJECT_ROOT / "data" / "splits"
    output_dir = PROJECT_ROOT / "artifacts" / "prism"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load splits
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")

    if args.nrows:
        train_df = train_df.head(args.nrows)
        val_df = val_df.head(args.nrows // 4)
        test_df = test_df.head(args.nrows // 4)

    logger.info(f"Loaded splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Extract structural features
    logger.info("Extracting structural features via FeaturePipeline...")
    feat_pipe = FeaturePipeline(normalization="standard")
    X_train_struct = feat_pipe.fit_transform(train_df)
    X_val_struct = feat_pipe.transform(val_df)
    X_test_struct = feat_pipe.transform(test_df)
    num_features = X_train_struct.shape[1]

    # Map sources to integers
    unique_sources = sorted(list(train_df["dataset_id"].unique()))
    source_to_id = {s: i for i, s in enumerate(unique_sources)}
    train_source_ids = [source_to_id.get(s, 0) for s in train_df["dataset_id"]]
    val_source_ids = [source_to_id.get(s, 0) for s in val_df["dataset_id"]]
    test_source_ids = [source_to_id.get(s, 0) for s in test_df["dataset_id"]]

    # Save feature pipeline
    joblib.dump(feat_pipe, output_dir / "feature_pipeline.joblib")

    # 3. Datasets & Loaders
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_dataset = PRISMDataset(prepare_text(train_df), X_train_struct, train_df["label"].tolist(), train_source_ids, tokenizer, max_length=args.max_length)
    val_dataset = PRISMDataset(prepare_text(val_df), X_val_struct, val_df["label"].tolist(), val_source_ids, tokenizer, max_length=args.max_length)
    test_dataset = PRISMDataset(prepare_text(test_df), X_test_struct, test_df["label"].tolist(), test_source_ids, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: prism_collate_fn(b, tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, collate_fn=lambda b: prism_collate_fn(b, tokenizer))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False, collate_fn=lambda b: prism_collate_fn(b, tokenizer))

    # 4. PRISM Model
    logger.info(f"Initializing PRISM-Phish with {num_features} structural features and {len(unique_sources)} sources")
    model = PRISMPhish(
        model_name=args.model,
        num_structural_features=num_features,
        num_sources=len(unique_sources),
        domain_adversarial=args.grl,
    )
    model.to(device)

    criterion_phish = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # 5. Training loop
    best_val_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            struct_features = batch["struct_features"].to(device)
            labels = batch["labels"].to(device)
            source_ids = batch["source_ids"].to(device)

            # Update GRL progress
            current_step = step + (epoch - 1) * len(train_loader)
            if args.grl:
                model.update_grl_lambda(current_step, total_steps)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                structural_features=struct_features,
                source_labels=source_ids if args.grl else None,
                labels=labels,
            )
            loss_phish = outputs.get("phishing_loss", criterion_phish(outputs["logits"], labels))

            loss = loss_phish
            if args.grl and "domain_loss" in outputs and outputs["domain_loss"] is not None:
                loss = loss + args.grl_lambda * outputs["domain_loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                logger.info(f"Epoch {epoch}/{args.epochs} | Step {step+1}/{len(train_loader)} | Loss: {total_loss / (step + 1):.4f}")

        val_metrics = evaluate_prism(model, val_loader, device)
        logger.info(f"Epoch {epoch} Val F1: {val_metrics['test_f1']:.4f}, ROC-AUC: {val_metrics['test_roc_auc']:.4f}")

        if val_metrics["test_f1"] > best_val_f1:
            best_val_f1 = val_metrics["test_f1"]
            torch.save(model.state_dict(), output_dir / "best_prism_model.pt")

    # Final evaluation
    logger.info("=== Evaluating PRISM-Phish on Test Set ===")
    test_metrics = evaluate_prism(model, test_loader, device)
    logger.info(f"PRISM Test ROC-AUC: {test_metrics['test_roc_auc']:.4f}, F1: {test_metrics['test_f1']:.4f}")

    with open(output_dir / "prism_results.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Saved PRISM results to {output_dir / 'prism_results.json'}")


if __name__ == "__main__":
    main()
