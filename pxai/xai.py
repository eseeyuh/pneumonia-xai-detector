"""Grad-CAM / Grad-CAM++ and quantitative evaluation of saliency maps.

The CAMs are computed functionally (``torch.autograd.grad`` on the feature map)
instead of with module hooks, so they are thread-safe on a shared model.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F


def _feature_grads(model, x: torch.Tensor, target: int | torch.Tensor | None):
    x = x.detach()
    with torch.no_grad():
        fmap = model.features(x)  # (N, C, h, w), output of norm5 (the CAM target layer)
    with torch.enable_grad():
        fmap = fmap.requires_grad_(True)
        logits = model.head(model.pool(fmap))
        if target is None:
            target = logits.argmax(dim=1)
        target = torch.as_tensor(target, device=logits.device).reshape(-1)
        if target.numel() == 1 and logits.shape[0] > 1:
            target = target.expand(logits.shape[0])
        score = logits.gather(1, target.view(-1, 1)).sum()
        (grads,) = torch.autograd.grad(score, fmap)
    return fmap.detach(), grads.detach(), logits.detach()


def _normalize(cam: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    cam = F.interpolate(cam.unsqueeze(1), size=size, mode="bilinear", align_corners=False)[:, 0]
    flat = cam.flatten(1)
    lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
    cam = ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(cam)
    return cam.cpu().numpy()


def grad_cam(model, x: torch.Tensor, target: int | None = None) -> np.ndarray:
    """Grad-CAM (Selvaraju et al., 2017). Returns (N, H, W) maps in [0, 1]."""
    fmap, grads, _ = _feature_grads(model, x, target)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * fmap).sum(1))
    return _normalize(cam, tuple(x.shape[-2:]))


def grad_cam_pp(model, x: torch.Tensor, target: int | None = None) -> np.ndarray:
    """Grad-CAM++ (Chattopadhyay et al., 2018), exponential-score approximation."""
    fmap, grads, _ = _feature_grads(model, x, target)
    g2, g3 = grads.pow(2), grads.pow(3)
    denom = 2 * g2 + (fmap * g3).sum(dim=(2, 3), keepdim=True)
    alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
    weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * fmap).sum(1))
    return _normalize(cam, tuple(x.shape[-2:]))


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45, cmap: str = "jet") -> np.ndarray:
    """Blend a [0,1] heat map onto a [0,1] RGB image; returns uint8 RGB."""
    from matplotlib import colormaps

    heat = colormaps[cmap](cam)[..., :3]
    out = (1 - alpha) * rgb + alpha * heat
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- localisation metrics
def boxes_to_mask(
    boxes: list[tuple[float, float, float, float]], shape: tuple[int, int]
) -> np.ndarray:
    """(x, y, w, h) boxes in pixel coordinates of ``shape`` → boolean mask."""
    mask = np.zeros(shape, dtype=bool)
    for x, y, w, h in boxes:
        x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
        x1, y1 = min(shape[1], int(np.ceil(x + w))), min(shape[0], int(np.ceil(y + h)))
        mask[y0:y1, x0:x1] = True
    return mask


def pointing_game(cam: np.ndarray, mask: np.ndarray, tolerance: int = 0) -> bool:
    """Hit if the CAM maximum falls inside the (dilated) ground-truth region."""
    yx = np.unravel_index(np.argmax(cam), cam.shape)
    if tolerance:
        y, x = yx
        return bool(
            mask[
                max(0, y - tolerance) : y + tolerance + 1, max(0, x - tolerance) : x + tolerance + 1
            ].any()
        )
    return bool(mask[yx])


def energy_in_mask(cam: np.ndarray, mask: np.ndarray) -> float:
    """Energy-based pointing game (Wang et al., 2020): share of CAM mass inside the mask."""
    total = cam.sum()
    return float(cam[mask].sum() / total) if total > 0 else float("nan")


def iou_at_threshold(cam: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float:
    pred = cam >= threshold
    union = (pred | mask).sum()
    return float((pred & mask).sum() / union) if union else float("nan")


# --------------------------------------------------------------------------- sanity check
def randomize_head_and_last_block(model, seed: int = 0):
    """Copy of the model with the classifier and last dense block re-initialised.

    Used for the model-parameter randomisation test (Adebayo et al., 2018): a
    faithful explanation should change substantially after randomisation.
    """
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    for module in list(m.classifier.modules()) + list(m.features.denseblock4.modules()):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return m.eval()


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation between two saliency maps."""
    ra = np.argsort(np.argsort(a.ravel())).astype(float)
    rb = np.argsort(np.argsort(b.ravel())).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")
