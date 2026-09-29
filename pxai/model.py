"""DenseNet-121 classifier with a split between the backbone and the head.

Splitting ``embed`` (backbone + pooling) from ``head`` (Dropout + Linear) lets
MC-Dropout reuse one backbone pass for all stochastic samples. Because the only
Dropout lives in the head, this is mathematically identical to T full forward
passes, but T times cheaper and free of shared-state side effects.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121

HF_REPO_ID = "eseeyuh/pneumonia-xai-detector"
HF_FILENAME = "best.pt"


class PneumoniaNet(nn.Module):
    """DenseNet-121 backbone + ``Dropout(p) -> Linear(1024, n_classes)`` head.

    The state-dict layout (``features.*``, ``classifier.1.*``) is identical to
    torchvision's DenseNet with a replaced classifier, so checkpoints trained with
    the original notebook load unchanged.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3, pretrained: bool = False):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        base = densenet121(weights=weights)
        self.features = base.features
        in_feats = base.classifier.in_features
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_feats, num_classes))
        self.dropout_p = dropout

    # -- pieces ---------------------------------------------------------------
    def pool(self, fmap: torch.Tensor) -> torch.Tensor:
        """Same post-processing torchvision's DenseNet applies after ``features``."""
        return torch.flatten(F.adaptive_avg_pool2d(F.relu(fmap), (1, 1)), 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x))

    def head(self, z: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
        """Apply the classifier. ``stochastic=True`` samples a dropout mask
        regardless of the module's mode (stateless MC-Dropout)."""
        if stochastic:
            return self.classifier[1](F.dropout(z, p=self.dropout_p, training=True))
        return self.classifier(z)  # Dropout follows its own train/eval flag, as in torchvision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


def build_model(
    num_classes: int = 2, dropout: float = 0.3, pretrained: bool = True
) -> PneumoniaNet:
    return PneumoniaNet(num_classes=num_classes, dropout=dropout, pretrained=pretrained)


def _extract_state_dict(obj) -> dict:
    """Accept a bare state dict or a training checkpoint dict."""
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def load_model(
    path: str | Path,
    device: torch.device | str = "cpu",
    dropout: float = 0.3,
    num_classes: int = 2,
) -> PneumoniaNet:
    model = PneumoniaNet(num_classes=num_classes, dropout=dropout, pretrained=False)
    obj = torch.load(path, map_location=device, weights_only=True)
    state = _extract_state_dict(obj)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def download_weights(
    repo_id: str = HF_REPO_ID,
    filename: str = HF_FILENAME,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """Download weights from the Hugging Face Hub (cached, atomic, resumable).

    Pin ``revision`` to a commit hash for reproducible deployments.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id, filename=filename, revision=revision, cache_dir=cache_dir
    )
