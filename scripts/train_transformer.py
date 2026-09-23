#!/usr/bin/env python3
"""
Train Transformer Baseline — Stage 4

Fine-tunes text-only DeBERTa-v3-small (or DistilBERT/RoBERTa fallback) on email text.
Evaluates on test split (ROC-AUC, PR-AUC, F1, Accuracy, FPR@99% Recall).
Saves model checkpoints, tokenizer, and metrics to artifacts/transformer/.

Usage:
    python scripts/train_transformer.py [--model microsoft/deberta-v3-small] [--epochs 3] [--batch-size 16]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_config
from src.evaluation.metrics import compute_all_metrics
from src.training.seeds import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_transformer")


class EmailDataset(Dataset):
    """PyTorch Dataset for Tokenized Emails."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,  # dynamically padded in collator
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def collate_fn(batch, tokenizer):
    """Dynamically pad batch to max length in batch."""
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def prepare_text(df: pd.DataFrame) -> list[str]:
    """Combine subject and body."""
    subj = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    return (subj + " " + body).tolist()


def evaluate_model(model, dataloader, device):
    """Compute predictions and evaluate metrics."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return compute_all_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        prefix="test",
    )


def main():
    parser = argparse.ArgumentParser(description="Train transformer baseline")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-small", help="HF model name")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max token sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows for rapid testing")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    splits_dir = PROJECT_ROOT / "data" / "splits"
    output_dir = PROJECT_ROOT / "artifacts" / "transformer"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")

    if args.nrows:
        train_df = train_df.head(args.nrows)
        val_df = val_df.head(args.nrows // 4)
        test_df = test_df.head(args.nrows // 4)

    logger.info(f"Loaded splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Tokenizer & Datasets
    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataset = EmailDataset(prepare_text(train_df), train_df["label"].tolist(), tokenizer, max_length=args.max_length)
    val_dataset = EmailDataset(prepare_text(val_df), val_df["label"].tolist(), tokenizer, max_length=args.max_length)
    test_dataset = EmailDataset(prepare_text(test_df), test_df["label"].tolist(), tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # 3. Model & Optimizer
    logger.info(f"Loading sequence classification model: {args.model}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # 4. Training Loop
    logger.info("=== Starting Transformer Fine-Tuning ===")
    best_val_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                logger.info(f"Epoch {epoch}/{args.epochs} | Step {step+1}/{len(train_loader)} | Loss: {total_loss / (step + 1):.4f}")

        val_metrics = evaluate_model(model, val_loader, device)
        logger.info(f"Epoch {epoch} Val F1: {val_metrics['test_f1']:.4f}, ROC-AUC: {val_metrics['test_roc_auc']:.4f}")

        if val_metrics["test_f1"] > best_val_f1:
            best_val_f1 = val_metrics["test_f1"]
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")

    # 5. Final Test Evaluation
    logger.info("=== Evaluating on Test Set ===")
    test_metrics = evaluate_model(model, test_loader, device)
    logger.info(f"Test ROC-AUC: {test_metrics['test_roc_auc']:.4f}, Test F1: {test_metrics['test_f1']:.4f}")

    results_path = output_dir / "transformer_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Saved test metrics to {results_path}")


if __name__ == "__main__":
    main()
