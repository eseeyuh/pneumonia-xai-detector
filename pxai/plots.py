"""Publication figures (matplotlib, vector-friendly)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pxai.metrics import expected_calibration_error, risk_coverage

PALETTE = ["#1f5f99", "#d1495b", "#1f9d8b", "#e0a458", "#6c5b7b", "#5c6b7a", "#2a9d8f", "#8d6a9f"]


def _roc_points(y, s):
    order = np.argsort(-s, kind="mergesort")
    y = np.asarray(y)[order].astype(bool)
    tpr = np.concatenate([[0], np.cumsum(y) / max(1, y.sum())])
    fpr = np.concatenate([[0], np.cumsum(~y) / max(1, (~y).sum())])
    return fpr, tpr


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, op=None) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for (name, (y, s)), c in zip(curves.items(), PALETTE, strict=False):
        fpr, tpr = _roc_points(y, s)
        ax.plot(fpr, tpr, color=c, lw=1.8, label=name)
    if op is not None:
        ax.scatter([1 - op[1]], [op[0]], color="black", zorder=5, s=25, label="operating point")
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    ax.set(xlabel="1 − specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _save(fig, path)


def plot_reliability(y, probs_by_name: dict[str, np.ndarray], path: Path, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0.5, 1], [0.5, 1], ls="--", color="#999", lw=1)
    for (name, probs), c in zip(probs_by_name.items(), PALETTE, strict=False):
        ece, d = expected_calibration_error(y, probs, n_bins)
        m = d["count"] > 0
        ax.plot(
            d["confidence"][m],
            d["accuracy"][m],
            "o-",
            color=c,
            ms=4,
            label=f"{name} (ECE {ece:.3f})",
        )
    ax.set(xlabel="Confidence", ylabel="Accuracy", xlim=(0.5, 1.0), ylim=(0.5, 1.01))
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def plot_risk_coverage(y, pred, scores: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for (name, u), c in zip(scores.items(), PALETTE, strict=False):
        cov, risk = risk_coverage(y, pred, u)
        ax.plot(cov, risk, color=c, lw=1.6, label=name)
    errors = (np.asarray(pred) != np.asarray(y)).astype(float)
    cov, risk = risk_coverage(y, pred, errors)
    ax.plot(cov, risk, color="black", ls=":", lw=1.2, label="oracle")
    ax.set(
        xlabel="Coverage (fraction of cases decided automatically)",
        ylabel="Risk (error rate)",
        xlim=(0, 1),
        ylim=(0, None),
    )
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, path)


def _save(fig, path: Path) -> None:
    for side in ("top", "right"):
        fig.axes[0].spines[side].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
