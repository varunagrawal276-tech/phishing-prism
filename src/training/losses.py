"""
Training losses for PRISM-Phish.

Combined loss: L_total = L_phishing + λ_domain * L_domain + λ_consistency * L_consistency
All λ values are configurable.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.consistency import ConsistencyLoss

logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """
    Combined training loss for PRISM-Phish.

    L_total = L_phishing + λ_domain * L_domain + λ_consistency * L_consistency
    """

    def __init__(
        self,
        lambda_domain: float = 0.1,
        lambda_consistency: float = 0.5,
        consistency_type: str = "symmetric_kl",
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.lambda_domain = lambda_domain
        self.lambda_consistency = lambda_consistency

        # Phishing classification loss (with optional class weights)
        self.phishing_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Domain classification loss
        self.domain_loss_fn = nn.CrossEntropyLoss()

        # Consistency loss
        self.consistency_loss = ConsistencyLoss(loss_type=consistency_type)

        logger.info(
            f"CombinedLoss: λ_domain={lambda_domain}, "
            f"λ_consistency={lambda_consistency}, "
            f"consistency={consistency_type}"
        )

    def forward(
        self,
        phishing_logits: torch.Tensor,
        labels: torch.Tensor,
        domain_logits: Optional[torch.Tensor] = None,
        source_labels: Optional[torch.Tensor] = None,
        probs_original: Optional[torch.Tensor] = None,
        probs_perturbed: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute combined loss.

        Returns dict with individual and total loss components.
        """
        losses = {}

        # Phishing classification loss (always present)
        losses["phishing"] = self.phishing_loss_fn(phishing_logits, labels)
        total = losses["phishing"]

        # Domain adversarial loss (optional)
        if domain_logits is not None and source_labels is not None and self.lambda_domain > 0:
            losses["domain"] = self.domain_loss_fn(domain_logits, source_labels)
            total = total + self.lambda_domain * losses["domain"]

        # Consistency loss (optional)
        if probs_original is not None and probs_perturbed is not None and self.lambda_consistency > 0:
            losses["consistency"] = self.consistency_loss(probs_original, probs_perturbed)
            total = total + self.lambda_consistency * losses["consistency"]

        losses["total"] = total
        return losses
