"""Uncertainty estimates for a binary classifier.

All scores follow the convention *higher = less certain*, so they can be
plugged straight into risk-coverage analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-12


def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in nats (max = ln 2 ≈ 0.693 for two classes)."""
    p = np.clip(probs, EPS, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


@dataclass
class MCResult:
    samples: np.ndarray  # (N, T, C) softmax samples
    mean_probs: np.ndarray  # (N, C)

    @property
    def predictive_entropy(self) -> np.ndarray:
        """Total uncertainty H[E_q p(y|x,w)]."""
        return entropy(self.mean_probs)

    @property
    def expected_entropy(self) -> np.ndarray:
        """Aleatoric part E_q H[p(y|x,w)]."""
        return entropy(self.samples).mean(axis=1)

    @property
    def mutual_information(self) -> np.ndarray:
        """Epistemic part (BALD): predictive entropy − expected entropy."""
        return np.clip(self.predictive_entropy - self.expected_entropy, 0.0, None)

    @property
    def variance(self) -> np.ndarray:
        """Variance of P(positive class) across samples."""
        return self.samples[:, :, 1].var(axis=1)


@torch.no_grad()
def mc_dropout(model, x: torch.Tensor, n_samples: int = 30, temperature: float = 1.0) -> MCResult:
    """Stateless MC-Dropout: one backbone pass, ``n_samples`` stochastic head passes.

    The model is never switched to train mode, so a shared (e.g. cached) model
    is safe to use from several threads.
    """
    z = model.embed(x)  # (N, D)
    zr = z.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, z.shape[1])  # (N*T, D)
    logits = model.head(zr, stochastic=True) / temperature
    samples = F.softmax(logits, dim=1).reshape(z.shape[0], n_samples, -1).cpu().numpy()
    return MCResult(samples=samples, mean_probs=samples.mean(axis=1))


@torch.no_grad()
def tta_probs(model, pil_images, transform, n_samples: int = 16, device="cpu") -> np.ndarray:
    """Test-time augmentation: returns (N, T, C) softmax samples."""
    out = []
    for img in pil_images:
        batch = torch.stack([transform(img) for _ in range(n_samples)]).to(device)
        out.append(F.softmax(model(batch), dim=1).cpu().numpy())
    return np.stack(out)


def uncertainty_scores(
    probs: np.ndarray,
    mc: MCResult | None = None,
    ensemble_probs: np.ndarray | None = None,
    tta_samples: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Collect every uncertainty score we compare in the paper.

    Args:
        probs: (N, C) deterministic softmax.
        mc: MC-Dropout result.
        ensemble_probs: (N, M, C) softmax of M independently trained models.
        tta_samples: (N, T, C) softmax under test-time augmentation.
    """
    scores = {
        "msp": 1.0 - probs.max(axis=1),  # maximum softmax probability baseline
        "softmax_entropy": entropy(probs),
    }
    if mc is not None:
        scores["mc_entropy"] = mc.predictive_entropy
        scores["mc_mutual_info"] = mc.mutual_information
        scores["mc_variance"] = mc.variance
    if ensemble_probs is not None:
        ens = MCResult(samples=ensemble_probs, mean_probs=ensemble_probs.mean(axis=1))
        scores["ensemble_entropy"] = ens.predictive_entropy
        scores["ensemble_mutual_info"] = ens.mutual_information
    if tta_samples is not None:
        tta = MCResult(samples=tta_samples, mean_probs=tta_samples.mean(axis=1))
        scores["tta_entropy"] = tta.predictive_entropy
    return scores


# --------------------------------------------------------------------------- calibration
def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Temperature scaling (Guo et al., 2017), fitted on the *validation* set."""
    lg = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(np.array(labels), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
