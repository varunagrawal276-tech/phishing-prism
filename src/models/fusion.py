"""
Gated fusion module for PRISM-Phish.

Combines contextual text representations with structural security features
using a lightweight gated fusion mechanism.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StructuralMLP(nn.Module):
    """MLP encoder for structural/security features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = None,
        output_dim: int = 128,
        dropout: float = 0.2,
        activation: str = "gelu",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GatedFusion(nn.Module):
    """
    Gated fusion of text and structural representations.

    g = sigmoid(W[h_text; h_struct])
    h_fused = g * h_text + (1 - g) * projection(h_struct)
    """

    def __init__(
        self,
        text_dim: int,
        struct_dim: int,
        output_dim: int = 256,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.struct_dim = struct_dim
        self.output_dim = output_dim

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(text_dim + struct_dim, output_dim),
            nn.Sigmoid(),
        )

        # Projection to align dimensions
        self.text_proj = nn.Linear(text_dim, output_dim) if text_dim != output_dim else nn.Identity()
        self.struct_proj = nn.Linear(struct_dim, output_dim) if struct_dim != output_dim else nn.Identity()

        logger.info(
            f"GatedFusion: text_dim={text_dim}, struct_dim={struct_dim}, "
            f"output_dim={output_dim}"
        )

    def forward(
        self,
        h_text: torch.Tensor,
        h_struct: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse text and structural representations.

        Args:
            h_text: (batch, text_dim) — from Transformer encoder
            h_struct: (batch, struct_dim) — from structural MLP

        Returns:
            h_fused: (batch, output_dim)
        """
        # Compute gate
        combined = torch.cat([h_text, h_struct], dim=-1)
        g = self.gate(combined)

        # Project to common dimension
        h_text_proj = self.text_proj(h_text)
        h_struct_proj = self.struct_proj(h_struct)

        # Gated combination
        h_fused = g * h_text_proj + (1 - g) * h_struct_proj

        return h_fused
