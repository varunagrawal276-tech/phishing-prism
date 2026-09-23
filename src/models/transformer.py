"""
Transformer model for PRISM-Phish.

DeBERTa-v3-small text-only classifier and base encoder for PRISM-Phish.
Supports mixed precision, gradient accumulation, and dynamic padding.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TransformerClassifier(nn.Module):
    """
    Text-only Transformer classifier.

    Uses DeBERTa-v3-small (or fallback) as encoder with a classification head.
    Serves as both a baseline and the text encoder component of PRISM-Phish.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        num_labels: int = 2,
        dropout: float = 0.1,
        pooling: str = "cls",
        freeze_embeddings: bool = False,
        freeze_n_layers: int = 0,
    ):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        self.model_name = model_name
        self.pooling = pooling
        self.num_labels = num_labels

        # Load pretrained encoder
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.hidden_size = self.config.hidden_size

        # Freeze layers if specified
        if freeze_embeddings:
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False

        if freeze_n_layers > 0:
            encoder_layers = getattr(self.encoder, 'encoder', None)
            if encoder_layers and hasattr(encoder_layers, 'layer'):
                for i, layer in enumerate(encoder_layers.layer):
                    if i < freeze_n_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                logger.info(f"Froze {freeze_n_layers} encoder layers")

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        logger.info(
            f"TransformerClassifier: {model_name}, "
            f"hidden={self.hidden_size}, pooling={pooling}"
        )

    def get_text_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract contextual text representation (h_text).

        Returns:
            Tensor of shape (batch_size, hidden_size).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)

        if self.pooling == "cls":
            return last_hidden[:, 0, :]  # [CLS] token
        elif self.pooling == "mean":
            # Mean pooling with attention mask
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            return summed / counts
        elif self.pooling == "max":
            mask = attention_mask.unsqueeze(-1).float()
            last_hidden = last_hidden.masked_fill(mask == 0, float("-inf"))
            return last_hidden.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Returns dict with:
            - logits: (batch, num_labels)
            - loss: scalar (if labels provided)
            - h_text: (batch, hidden_size)
            - probabilities: (batch, num_labels)
        """
        h_text = self.get_text_representation(input_ids, attention_mask)
        h_text_dropped = self.dropout(h_text)
        logits = self.classifier(h_text_dropped)
        probabilities = torch.softmax(logits, dim=-1)

        result = {
            "logits": logits,
            "h_text": h_text,
            "probabilities": probabilities,
        }

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    def count_parameters(self) -> dict:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
