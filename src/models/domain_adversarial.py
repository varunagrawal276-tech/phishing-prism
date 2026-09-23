"""
Domain-adversarial training for PRISM-Phish.

Implements Gradient Reversal Layer (GRL) and source classifier
to learn source-invariant representations.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import Function

logger = logging.getLogger(__name__)


class GradientReversalFunction(Function):
    """Gradient Reversal Layer — reverses gradient during backprop."""

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper for gradient reversal with scheduled lambda."""

    def __init__(self, lambda_val: float = 1.0):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_val)

    def set_lambda(self, lambda_val: float):
        self.lambda_val = lambda_val


class DomainClassifier(nn.Module):
    """
    Source/domain classifier with gradient reversal.

    Predicts which dataset an email came from.
    The GRL ensures the shared representation learns to be source-invariant.
    """

    def __init__(
        self,
        input_dim: int,
        num_sources: int = 7,
        hidden_dims: list[int] = None,
        dropout: float = 0.2,
        grl_lambda_max: float = 1.0,
        grl_schedule: str = "linear",
        grl_warmup_steps: int = 500,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.num_sources = num_sources
        self.grl_lambda_max = grl_lambda_max
        self.grl_schedule = grl_schedule
        self.grl_warmup_steps = grl_warmup_steps

        self.grl = GradientReversalLayer(lambda_val=0.0)

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, num_sources))
        self.classifier = nn.Sequential(*layers)

        logger.info(
            f"DomainClassifier: num_sources={num_sources}, "
            f"lambda_max={grl_lambda_max}, schedule={grl_schedule}"
        )

    def update_lambda(self, current_step: int, total_steps: int):
        """Update GRL lambda based on training progress."""
        if self.grl_schedule == "linear":
            progress = min(current_step / max(self.grl_warmup_steps, 1), 1.0)
            lambda_val = self.grl_lambda_max * progress
        elif self.grl_schedule == "exponential":
            progress = min(current_step / max(self.grl_warmup_steps, 1), 1.0)
            lambda_val = self.grl_lambda_max * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * progress)).item()) - 1.0)
        elif self.grl_schedule == "step":
            if current_step < self.grl_warmup_steps:
                lambda_val = 0.0
            else:
                lambda_val = self.grl_lambda_max
        else:
            lambda_val = self.grl_lambda_max

        self.grl.set_lambda(lambda_val)
        return lambda_val

    def forward(self, h_shared: torch.Tensor) -> torch.Tensor:
        """
        Predict source/dataset identity from shared representation.

        Args:
            h_shared: (batch, hidden_dim) — shared representation

        Returns:
            logits: (batch, num_sources) — source classification logits
        """
        reversed_h = self.grl(h_shared)
        return self.classifier(reversed_h)
