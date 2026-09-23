"""
Token Attribution for PRISM-Phish and Transformer Models.

Provides token-level saliency scores using input-gradient attribution
(x * grad) and occlusion to highlight words driving phishing vs legitimate decisions.
"""

import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TokenAttributionExplainer:
    """
    Computes token-level attribution scores for text sequences.
    """

    def __init__(self, model, tokenizer, device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def attribute_text(self, text: str, target_class: int = 1, max_length: int = 256) -> dict:
        """
        Compute token attributions using gradient of target class logit w.r.t embedding inputs.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

        # Determine embedding layer
        embedding_layer = None
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "embeddings"):
            embedding_layer = self.model.encoder.embeddings.word_embeddings
        elif hasattr(self.model, "distilbert") and hasattr(self.model.distilbert, "embeddings"):
            embedding_layer = self.model.distilbert.embeddings.word_embeddings
        elif hasattr(self.model, "text_encoder") and hasattr(self.model.text_encoder, "encoder"):
            enc = self.model.text_encoder.encoder
            if hasattr(enc, "embeddings"):
                embedding_layer = enc.embeddings.word_embeddings

        if embedding_layer is not None:
            # Gradient x Input method
            embeddings = embedding_layer(input_ids)
            embeddings.retain_grad()

            # Forward pass using hook or custom input
            # If standard huggingface sequence classification
            outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            target_score = logits[0, target_class]

            self.model.zero_grad()
            target_score.backward()

            grads = embeddings.grad[0]  # (seq_len, hidden_dim)
            # Dot product / norm
            scores = (embeddings[0] * grads).sum(dim=-1).detach().cpu().numpy()
        else:
            # Fallback: Occlusion-based attribution
            with torch.no_grad():
                base_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                base_logits = base_outputs.logits if hasattr(base_outputs, "logits") else base_outputs["logits"]
                base_prob = F.softmax(base_logits, dim=-1)[0, target_class].item()

            scores = []
            pad_token_id = self.tokenizer.pad_token_id or 0
            for i in range(input_ids.shape[1]):
                occluded_ids = input_ids.clone()
                occluded_ids[0, i] = pad_token_id
                with torch.no_grad():
                    occ_outputs = self.model(input_ids=occluded_ids, attention_mask=attention_mask)
                    occ_logits = occ_outputs.logits if hasattr(occ_outputs, "logits") else occ_outputs["logits"]
                    occ_prob = F.softmax(occ_logits, dim=-1)[0, target_class].item()
                scores.append(base_prob - occ_prob)
            scores = np.array(scores)

        # Normalize scores
        norm = np.linalg.norm(scores)
        if norm > 0:
            norm_scores = scores / norm
        else:
            norm_scores = scores

        # Package token-score pairs (filter special tokens)
        token_attributions = []
        for tok, score, raw_score in zip(tokens, norm_scores, scores):
            token_attributions.append({
                "token": tok,
                "importance": float(score),
                "raw_importance": float(raw_score),
            })

        # Rank tokens by absolute importance
        top_tokens = sorted(
            [t for t in token_attributions if t["token"] not in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"]],
            key=lambda x: abs(x["importance"]),
            reverse=True,
        )[:15]

        return {
            "tokens": token_attributions,
            "top_tokens": top_tokens,
            "prediction_class": int(target_class),
        }
