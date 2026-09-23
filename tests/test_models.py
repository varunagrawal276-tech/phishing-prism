import pytest
import torch

from src.models.consistency import ConsistencyLoss, whitespace_perturbation, casing_perturbation
from src.models.domain_adversarial import DomainClassifier, GradientReversalLayer
from src.models.fusion import GatedFusion, StructuralMLP


def test_structural_mlp():
    mlp = StructuralMLP(input_dim=20, hidden_dims=[32, 16], output_dim=16)
    x = torch.randn(4, 20)
    out = mlp(x)
    assert out.shape == (4, 16)


def test_gated_fusion():
    fusion = GatedFusion(text_dim=64, struct_dim=16, output_dim=32)
    h_text = torch.randn(4, 64)
    h_struct = torch.randn(4, 16)
    out = fusion(h_text, h_struct)
    assert out.shape == (4, 32)


def test_domain_classifier_and_grl():
    domain_clf = DomainClassifier(input_dim=32, num_sources=5, hidden_dims=[16])
    domain_clf.update_lambda(current_step=250, total_steps=500)
    x = torch.randn(4, 32)
    logits = domain_clf(x)
    assert logits.shape == (4, 5)


def test_consistency_loss():
    loss_fn = ConsistencyLoss(loss_type="symmetric_kl")
    p = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    q = torch.tensor([[0.75, 0.25], [0.15, 0.85]])
    loss = loss_fn(p, q)
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0.0


def test_perturbation_functions():
    text = "Please verify your bank account credentials immediately."
    ws = whitespace_perturbation(text, intensity=2)
    assert len(ws) >= len(text)
    case = casing_perturbation(text, intensity=2)
    assert len(case) == len(text)

