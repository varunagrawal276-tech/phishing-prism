"""
Perturbation consistency training for PRISM-Phish.

Implements label-preserving transformations and consistency loss
to improve prediction stability under controlled perturbations.
"""

import logging
import random
import re
import string
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =====================================================
# Label-Preserving Transformations
# =====================================================

def whitespace_perturbation(text: str, intensity: int = 1) -> str:
    """Add/modify whitespace without changing semantics."""
    if not text:
        return text
    result = text
    for _ in range(intensity):
        pos = random.randint(0, max(len(result) - 1, 0))
        result = result[:pos] + " " + result[pos:]
    return result


def casing_perturbation(text: str, intensity: int = 1) -> str:
    """Randomly change case of some characters."""
    if not text:
        return text
    chars = list(text)
    n_changes = min(intensity * 2, len(chars))
    positions = random.sample(range(len(chars)), min(n_changes, len(chars)))
    for pos in positions:
        if chars[pos].isalpha():
            chars[pos] = chars[pos].swapcase()
    return "".join(chars)


def punctuation_perturbation(text: str, intensity: int = 1) -> str:
    """Minor punctuation changes (add/remove harmless punctuation)."""
    if not text:
        return text
    result = text
    for _ in range(intensity):
        r = random.random()
        if r < 0.5 and len(result) > 0:
            # Add a period or comma at a random space
            spaces = [i for i, c in enumerate(result) if c == " "]
            if spaces:
                pos = random.choice(spaces)
                punct = random.choice([".", ","])
                result = result[:pos] + punct + result[pos:]
        else:
            # Remove a trailing period or comma
            result = result.rstrip(".,") + result[-1:] if result.endswith((".", ",")) else result
    return result


def character_perturbation(text: str, intensity: int = 1) -> str:
    """Minor character-level changes (typos)."""
    if not text or len(text) < 3:
        return text
    chars = list(text)
    n_changes = min(intensity, max(1, len(chars) // 50))
    for _ in range(n_changes):
        pos = random.randint(0, len(chars) - 1)
        if chars[pos].isalpha():
            # Swap adjacent characters
            if pos < len(chars) - 1:
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


PERTURBATION_FUNCTIONS = {
    "whitespace": whitespace_perturbation,
    "casing": casing_perturbation,
    "punctuation": punctuation_perturbation,
    "character": character_perturbation,
}


def apply_random_perturbation(
    text: str,
    intensity: int = 1,
    perturbation_types: Optional[list[str]] = None,
) -> str:
    """Apply a random label-preserving perturbation to text."""
    if perturbation_types is None:
        perturbation_types = list(PERTURBATION_FUNCTIONS.keys())

    ptype = random.choice(perturbation_types)
    fn = PERTURBATION_FUNCTIONS[ptype]
    return fn(text, intensity=intensity)


# =====================================================
# Consistency Loss
# =====================================================

def symmetric_kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Symmetric KL divergence between two probability distributions.

    D_SKL(p, q) = 0.5 * (KL(p || q) + KL(q || p))
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    kl_pq = F.kl_div(q.log(), p, reduction="batchmean")
    kl_qp = F.kl_div(p.log(), q, reduction="batchmean")
    return 0.5 * (kl_pq + kl_qp)


def js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Jensen-Shannon divergence between two distributions."""
    m = 0.5 * (p + q)
    return 0.5 * (
        F.kl_div(m.clamp(min=eps).log(), p.clamp(min=eps), reduction="batchmean")
        + F.kl_div(m.clamp(min=eps).log(), q.clamp(min=eps), reduction="batchmean")
    )


class ConsistencyLoss(nn.Module):
    """
    Consistency loss between original and perturbed predictions.

    Encourages the model to produce similar predictions for
    an email and its label-preserving perturbation.
    """

    def __init__(self, loss_type: str = "symmetric_kl"):
        super().__init__()
        self.loss_type = loss_type

        self.loss_fn = {
            "symmetric_kl": symmetric_kl_divergence,
            "js_divergence": js_divergence,
            "mse": lambda p, q: F.mse_loss(p, q),
        }[loss_type]

        logger.info(f"ConsistencyLoss: {loss_type}")

    def forward(
        self,
        probs_original: torch.Tensor,
        probs_perturbed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute consistency loss.

        Args:
            probs_original: (batch, num_classes) — predictions on original input
            probs_perturbed: (batch, num_classes) — predictions on perturbed input

        Returns:
            Scalar consistency loss.
        """
        return self.loss_fn(probs_original, probs_perturbed)
