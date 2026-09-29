import numpy as np
import pytest

from pxai.metrics import (
    accuracy_at_coverage,
    aurc,
    average_precision,
    bootstrap_ci,
    expected_calibration_error,
    oracle_aurc,
    risk_coverage,
    roc_auc,
    select_threshold,
    threshold_metrics,
)

sk = pytest.importorskip("sklearn.metrics")
rng = np.random.default_rng(0)
Y = rng.integers(0, 2, 500)
S = np.clip(Y * 0.3 + rng.normal(0.4, 0.2, 500), 0, 1)


def test_auc_matches_sklearn():
    assert roc_auc(Y, S) == pytest.approx(sk.roc_auc_score(Y, S), abs=1e-10)
    s_ties = np.round(S, 1)
    assert roc_auc(Y, s_ties) == pytest.approx(sk.roc_auc_score(Y, s_ties), abs=1e-10)


def test_average_precision_matches_sklearn():
    assert average_precision(Y, S) == pytest.approx(sk.average_precision_score(Y, S), abs=1e-10)


def test_threshold_metrics_against_sklearn():
    m = threshold_metrics(Y, S, 0.5)
    pred = S >= 0.5
    assert m["sensitivity"] == pytest.approx(sk.recall_score(Y, pred))
    assert m["precision"] == pytest.approx(sk.precision_score(Y, pred))
    assert m["f1"] == pytest.approx(sk.f1_score(Y, pred))
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == len(Y)


def test_select_threshold_reaches_target_sensitivity():
    t = select_threshold(Y, S, target_sensitivity=0.95)
    assert threshold_metrics(Y, S, t)["sensitivity"] >= 0.95
    # and it is the *highest* such threshold
    higher = S[S > t]
    if len(higher):
        assert threshold_metrics(Y, S, higher.min())["sensitivity"] < 0.95


def test_ece_zero_for_perfectly_calibrated_confident_model():
    y = np.array([0, 1, 1, 0])
    probs = np.eye(2)[y]
    assert expected_calibration_error(y, probs)[0] == pytest.approx(0.0)


def test_risk_coverage_and_aurc():
    y = np.array([0, 0, 1, 1, 1])
    pred = np.array([0, 1, 1, 0, 1])  # errors at idx 1 and 3
    good = np.array([0.1, 0.9, 0.2, 0.8, 0.3])  # errors ranked most uncertain
    bad = -good
    cov, risk = risk_coverage(y, pred, good)
    assert cov[-1] == 1.0 and risk[-1] == pytest.approx(0.4)
    assert risk[2] == 0.0  # three most certain are all correct
    assert aurc(y, pred, good) == pytest.approx(oracle_aurc(y, pred))
    assert aurc(y, pred, good) < aurc(y, pred, bad)
    assert accuracy_at_coverage(y, pred, good, 0.6) == 1.0


def test_bootstrap_ci_brackets_point():
    point, lo, hi = bootstrap_ci(roc_auc, Y, S, n_boot=300)
    assert lo <= point <= hi


def test_cluster_bootstrap_is_wider_with_duplicated_patients():
    # every "patient" duplicated 5 times: image-level bootstrap is overconfident
    y, s = np.repeat(Y[:100], 5), np.repeat(S[:100], 5)
    groups = np.repeat(np.arange(100), 5)
    _, lo_i, hi_i = bootstrap_ci(roc_auc, y, s, n_boot=400)
    _, lo_g, hi_g = bootstrap_ci(roc_auc, y, s, n_boot=400, groups=groups)
    assert (hi_g - lo_g) > (hi_i - lo_i)
