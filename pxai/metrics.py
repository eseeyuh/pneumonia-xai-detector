"""Evaluation metrics: discrimination, calibration, selective prediction.

Pure NumPy so it can be unit-tested and reused without scikit-learn.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


# --------------------------------------------------------------------------- discrimination
def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the Mann–Whitney U statistic (ties get half credit)."""
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    sorted_scores = score[order]
    # average ranks for ties
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision–recall curve (step-wise, tie-aware, as in scikit-learn)."""
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return float("nan")
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="mergesort")
    s, y_sorted = score[order], y[order]
    # evaluate only at the last index of each group of tied scores
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y_sorted)[last]
    precision = tp / (last + 1)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return {
        "tp": int((y & pred).sum()),
        "fp": int((~y & pred).sum()),
        "fn": int((y & ~pred).sum()),
        "tn": int((~y & ~pred).sum()),
    }


def threshold_metrics(y: np.ndarray, p_pos: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    c = confusion(y, p_pos >= threshold)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]

    def div(a, b):
        return a / b if b else float("nan")

    sens, spec, prec = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)
    return {
        "accuracy": div(tp + tn, tp + tn + fp + fn),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "npv": div(tn, tn + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (sens + spec) / 2,
        **{k: float(v) for k, v in c.items()},
    }


def select_threshold(
    y: np.ndarray, p_pos: np.ndarray, target_sensitivity: float | None = 0.95
) -> float:
    """Choose an operating point on the *validation* set.

    With ``target_sensitivity`` set, return the highest threshold reaching it
    (triage: missing pneumonia is costlier than a false alarm). Otherwise
    maximise Youden's J.
    """
    y = np.asarray(y).astype(bool)
    cands = np.unique(np.concatenate([p_pos, [0.0, 1.0]]))
    best_t, best_j = 0.5, -np.inf
    for t in cands[::-1]:  # high → low threshold, sensitivity increases
        m = threshold_metrics(y, p_pos, t)
        if target_sensitivity is not None:
            if m["sensitivity"] >= target_sensitivity:
                return float(t)
        else:
            j = m["sensitivity"] + m["specificity"] - 1
            if j > best_j:
                best_t, best_j = float(t), j
    return best_t if target_sensitivity is None else 0.0


# --------------------------------------------------------------------------- calibration
def expected_calibration_error(
    y: np.ndarray, probs: np.ndarray, n_bins: int = 15
) -> tuple[float, dict[str, np.ndarray]]:
    """Top-label ECE with equal-width bins. Returns (ece, reliability-diagram data)."""
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(y)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    acc = np.full(n_bins, np.nan)
    avg_conf = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        count[b] = m.sum()
        if count[b]:
            acc[b], avg_conf[b] = correct[m].mean(), conf[m].mean()
            ece += count[b] / len(conf) * abs(acc[b] - avg_conf[b])
    return float(ece), {"edges": edges, "accuracy": acc, "confidence": avg_conf, "count": count}


def brier_score(y: np.ndarray, p_pos: np.ndarray) -> float:
    return float(np.mean((np.asarray(p_pos) - np.asarray(y)) ** 2))


def nll(y: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), np.asarray(y)], 1e-12, 1.0)
    return float(-np.log(p).mean())


# --------------------------------------------------------------------------- selective prediction
def risk_coverage(
    y: np.ndarray, pred: np.ndarray, uncertainty: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Risk (error rate) of the retained set as the most uncertain cases are deferred.

    Returns (coverage, risk), both of length N, coverage ascending from 1/N to 1.
    """
    order = np.argsort(uncertainty, kind="mergesort")  # most certain first
    errors = (np.asarray(pred) != np.asarray(y))[order].astype(float)
    n = len(errors)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(errors) / np.arange(1, n + 1)
    return coverage, risk


def aurc(y: np.ndarray, pred: np.ndarray, uncertainty: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    _, risk = risk_coverage(y, pred, uncertainty)
    return float(risk.mean())


def oracle_aurc(y: np.ndarray, pred: np.ndarray) -> float:
    """AURC of a perfect ranker (defers all errors first); used for E-AURC."""
    errors = (np.asarray(pred) != np.asarray(y)).astype(float)
    return aurc(y, pred, errors)


def accuracy_at_coverage(
    y: np.ndarray, pred: np.ndarray, uncertainty: np.ndarray, coverage: float
) -> float:
    cov, risk = risk_coverage(y, pred, uncertainty)
    k = max(1, int(np.floor(coverage * len(cov))))
    return float(1.0 - risk[k - 1])


# --------------------------------------------------------------------------- confidence intervals
def bootstrap_ci(
    metric: Callable[..., float],
    *arrays: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (point, low, high).

    Pass ``groups`` (e.g. patient IDs) for a cluster bootstrap that resamples
    patients instead of images, which is the correct unit when a patient
    contributes several images.
    """
    rng = np.random.default_rng(seed)
    arrays = tuple(np.asarray(a) for a in arrays)
    point = metric(*arrays)
    n = len(arrays[0])
    if groups is not None:
        groups = np.asarray(groups)
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    stats = []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[g] for g in pick])
        with np.errstate(all="ignore"):
            v = metric(*(a[idx] for a in arrays))
        if np.isfinite(v):
            stats.append(v)
    if not stats:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)
