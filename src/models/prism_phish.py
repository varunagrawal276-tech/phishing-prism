"""
PRISM-Phish: Full model architecture.

Combines:
- DeBERTa-v3-small text encoder
- Structural/security feature MLP
- Gated fusion
- Domain-adversarial source classifier (with GRL)
- Phishing classifier head
- Perturbation consistency support
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from src.models.transformer import TransformerClassifier
from src.models.fusion import StructuralMLP, GatedFusion
from src.models.domain_adversarial import DomainClassifier

logger = logging.getLogger(__name__)


class PRISMPhish(nn.Module):
    """
    Full PRISM-Phish model.

    Architecture:
        Text → DeBERTa → h_text
                                  → GatedFusion → h_fused → Phishing Classifier
        Features → MLP → h_struct                         → Domain Classifier (GRL)
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        num_structural_features: int = 40,
        num_sources: int = 7,
        structural_hidden_dims: list[int] = None,
        fusion_dim: int = 256,
        classifier_hidden_dims: list[int] = None,
        classifier_dropout: float = 0.3,
        structural_dropout: float = 0.2,
        domain_adversarial: bool = True,
        grl_lambda_max: float = 1.0,
        grl_schedule: str = "linear",
        grl_warmup_steps: int = 500,
        pooling: str = "cls",
    ):
        super().__init__()

        if structural_hidden_dims is None:
            structural_hidden_dims = [128, 64]
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [128]

        # Text encoder
        self.text_encoder = TransformerClassifier(
            model_name=model_name,
            num_labels=2,  # Will use our own head
            pooling=pooling,
        )
        text_dim = self.text_encoder.hidden_size

        # Structural encoder
        struct_output_dim = structural_hidden_dims[-1] if structural_hidden_dims else 64
        self.structural_encoder = StructuralMLP(
            input_dim=num_structural_features,
            hidden_dims=structural_hidden_dims,
            output_dim=struct_output_dim,
            dropout=structural_dropout,
        )

        # Gated fusion
        self.fusion = GatedFusion(
            text_dim=text_dim,
            struct_dim=struct_output_dim,
            output_dim=fusion_dim,
        )

        # Phishing classifier head
        clf_layers = []
        prev_dim = fusion_dim
        for dim in classifier_hidden_dims:
            clf_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
            ])
            prev_dim = dim
        clf_layers.append(nn.Linear(prev_dim, 2))
        self.phishing_classifier = nn.Sequential(*clf_layers)

        # Domain adversarial branch (optional)
        self.domain_adversarial = domain_adversarial
        if domain_adversarial:
            self.domain_classifier = DomainClassifier(
                input_dim=fusion_dim,
                num_sources=num_sources,
                grl_lambda_max=grl_lambda_max,
                grl_schedule=grl_schedule,
                grl_warmup_steps=grl_warmup_steps,
            )

        param_counts = self.count_parameters()
        logger.info(
            f"PRISMPhish initialized: "
            f"{param_counts['total']:,} total params, "
            f"{param_counts['trainable']:,} trainable"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        structural_features: torch.Tensor,
        source_labels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Full forward pass.

        Returns dict with:
            - logits: phishing classification logits
            - probabilities: phishing probabilities
            - h_fused: fused representation
            - h_text: text representation
            - h_struct: structural representation
            - domain_logits: source classification logits (if enabled)
            - phishing_loss: phishing CE loss (if labels provided)
            - domain_loss: domain CE loss (if source_labels provided)
        """
        # Text encoding
        text_output = self.text_encoder.get_text_representation(input_ids, attention_mask)
        h_text = text_output

        # Structural encoding
        h_struct = self.structural_encoder(structural_features)

        # Fusion
        h_fused = self.fusion(h_text, h_struct)

        # Phishing classification
        logits = self.phishing_classifier(h_fused)
        probabilities = torch.softmax(logits, dim=-1)

        result = {
            "logits": logits,
            "probabilities": probabilities,
            "h_fused": h_fused,
            "h_text": h_text,
            "h_struct": h_struct,
        }

        # Phishing loss
        if labels is not None:
            result["phishing_loss"] = nn.CrossEntropyLoss()(logits, labels)

        # Domain adversarial loss
        if self.domain_adversarial and source_labels is not None:
            domain_logits = self.domain_classifier(h_fused)
            result["domain_logits"] = domain_logits
            result["domain_loss"] = nn.CrossEntropyLoss()(domain_logits, source_labels)

        return result

    def update_grl_lambda(self, current_step: int, total_steps: int) -> float:
        """Update the GRL coefficient based on training schedule."""
        if self.domain_adversarial:
            return self.domain_classifier.update_lambda(current_step, total_steps)
        return 0.0

    def count_parameters(self) -> dict:
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
