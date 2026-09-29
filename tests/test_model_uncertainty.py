import numpy as np
import pytest
import torch
import torch.nn as nn
from torchvision.models import densenet121

from pxai.model import PneumoniaNet, load_model
from pxai.uncertainty import MCResult, entropy, fit_temperature, mc_dropout, softmax_np

X = torch.randn(2, 3, 64, 64)


def test_state_dict_compatible_with_original_notebook_layout(tiny_model):
    """Weights trained with torchvision DenseNet + Sequential(Dropout, Linear) must load."""
    ref = densenet121(weights=None)
    ref.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(ref.classifier.in_features, 2))
    assert set(ref.state_dict()) == set(tiny_model.state_dict())
    ref.load_state_dict(tiny_model.state_dict())
    ref.eval()
    with torch.no_grad():
        assert torch.allclose(ref(X), tiny_model(X), atol=1e-5)


def test_load_model_accepts_checkpoint_dicts(tmp_path, tiny_model):
    p = tmp_path / "last.pt"
    torch.save({"model": {f"module.{k}": v for k, v in tiny_model.state_dict().items()}}, p)
    m = load_model(p)
    with torch.no_grad():
        assert torch.allclose(m(X), tiny_model(X), atol=1e-6)


def test_head_only_mc_equals_full_forward_with_dropout(tiny_model):
    """One stochastic head pass == a full forward pass with Dropout in train mode."""
    torch.manual_seed(123)
    fast = mc_dropout(tiny_model, X[:1], n_samples=1).samples[0, 0]
    ref = PneumoniaNet(pretrained=False)
    ref.load_state_dict(tiny_model.state_dict())
    ref.eval()
    ref.classifier[0].train()
    torch.manual_seed(123)
    with torch.no_grad():
        slow = torch.softmax(ref(X[:1]), 1)[0].numpy()
    assert np.allclose(fast, slow, atol=1e-5)


def test_mc_dropout_is_stateless(tiny_model):
    assert not tiny_model.training
    res = mc_dropout(tiny_model, X, n_samples=8)
    assert not any(m.training for m in tiny_model.modules())
    assert res.samples.shape == (2, 8, 2)
    assert np.allclose(res.samples.sum(-1), 1.0, atol=1e-5)
    assert res.samples[:, :, 1].std(axis=1).max() > 0  # samples actually differ


def test_deterministic_forward_is_deterministic(tiny_model):
    with torch.no_grad():
        assert torch.equal(tiny_model(X), tiny_model(X))


def test_uncertainty_decomposition():
    certain = np.tile([[0.99, 0.01]], (1, 10, 1))
    disagree = np.stack([np.tile([0.99, 0.01], (5, 1)), np.tile([0.01, 0.99], (5, 1))]).reshape(
        1, 10, 2
    )
    c, d = MCResult(certain, certain.mean(1)), MCResult(disagree, disagree.mean(1))
    assert c.mutual_information[0] == pytest.approx(0, abs=1e-9)
    assert d.mutual_information[0] > 0.5
    assert d.predictive_entropy[0] == pytest.approx(np.log(2), abs=1e-6)
    assert entropy(np.array([0.5, 0.5])) == pytest.approx(np.log(2))


def test_temperature_scaling_recovers_true_temperature():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, (4000, 2))
    y = np.array([rng.choice(2, p=p) for p in softmax_np(true_logits)])
    overconfident = true_logits * 3.0
    assert fit_temperature(overconfident, y) == pytest.approx(3.0, rel=0.1)
