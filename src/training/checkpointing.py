"""
Checkpointing utilities for PRISM-Phish.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    metrics: dict,
    path: Path,
    scheduler=None,
    extra_state: Optional[dict] = None,
) -> Path:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if extra_state:
        state.update(extra_state)

    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path}")

    # Also save metrics as JSON for easy reading
    meta_path = path.with_suffix(".json")
    meta = {
        "epoch": epoch,
        "step": step,
        "metrics": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in metrics.items()},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return path


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cpu",
) -> dict:
    """Load a training checkpoint."""
    logger.info(f"Loading checkpoint: {path}")

    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    logger.info(
        f"Loaded checkpoint: epoch={state.get('epoch')}, step={state.get('step')}"
    )
    return state
