"""Evaluate one model (or an ensemble) with everything the paper needs.

Internal evaluation (operating point, temperature and deferral threshold are
fitted on ``val`` and then frozen for ``test``)::

    python -m pxai.evaluate --manifest data/splits/kermany.csv --root data/chest_xray \\
        --checkpoints runs/seed42/best.pt --out results/kermany

Deep-ensemble comparison (first checkpoint = the reported single model)::

    python -m pxai.evaluate ... --checkpoints runs/seed{42,1,2,3,4}/best.pt

External validation with the operating point frozen from the internal run::

    python -m pxai.evaluate --manifest data/splits/rsna.csv --root data/rsna_png \\
        --checkpoints runs/seed42/best.pt --operating-point results/kermany/operating_point.json \\
        --out results/rsna

Outputs: ``predictions_<split>.csv``, ``metrics.json``, ``operating_point.json``,
``figures/*.png|pdf``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pxai.data import XrayDataset, read_manifest
from pxai.metrics import (
    accuracy_at_coverage,
    aurc,
    average_precision,
    bootstrap_ci,
    brier_score,
    expected_calibration_error,
    nll,
    oracle_aurc,
    roc_auc,
    select_threshold,
    threshold_metrics,
)
from pxai.model import load_model
from pxai.plots import plot_reliability, plot_risk_coverage, plot_roc
from pxai.preprocessing import eval_transform, tta_transform
from pxai.uncertainty import (
    MCResult,
    fit_temperature,
    mc_dropout,
    softmax_np,
    uncertainty_scores,
)
from pxai.utils import get_device, seed_everything

COVERAGES = (0.7, 0.8, 0.9)


@torch.no_grad()
def collect(
    model, dataset, device, mc_samples: int, batch_size: int = 64, workers: int = 4
) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    logits, mc = [], []
    model.eval()
    for x, _, _ in loader:
        x = x.to(device)
        logits.append(model(x).float().cpu().numpy())
        if mc_samples:
            mc.append(mc_dropout(model, x, n_samples=mc_samples).samples)
    return {"logits": np.concatenate(logits), "mc": np.concatenate(mc) if mc else None}


@torch.no_grad()
def collect_tta(
    model, df, root, device, n: int, size: int, mode: str, workers: int = 4
) -> np.ndarray:
    ds = XrayDataset(df, root, tta_transform(size, mode))
    out = []
    for _ in range(n):
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=workers)
        out.append(
            np.concatenate(
                [torch.softmax(model(x.to(device)), 1).cpu().numpy() for x, _, _ in loader]
            )
        )
    return np.stack(out, axis=1)  # (N, T, C)


def with_ci(metric, *arrays, groups=None, n_boot=2000):
    point, lo, hi = bootstrap_ci(metric, *arrays, groups=groups, n_boot=n_boot)
    return {"value": point, "ci95": [lo, hi]}


def evaluate_split(y, probs, mc, scores, threshold, groups, n_boot) -> dict:
    p = probs[:, 1]
    pred = (p >= threshold).astype(int)
    res: dict = {
        "n_images": int(len(y)),
        "n_patients": int(len(np.unique(groups))) if groups is not None else None,
        "prevalence": float(np.mean(y)),
        "auc": with_ci(roc_auc, y, p, groups=groups, n_boot=n_boot),
        "average_precision": with_ci(average_precision, y, p, groups=groups, n_boot=n_boot),
        "threshold": threshold,
    }
    for name in (
        "sensitivity",
        "specificity",
        "precision",
        "npv",
        "f1",
        "accuracy",
        "balanced_accuracy",
    ):
        res[name] = with_ci(
            lambda a, b, k=name: threshold_metrics(a, b, threshold)[k],
            y,
            p,
            groups=groups,
            n_boot=n_boot,
        )
    res["confusion"] = {
        k: int(v)
        for k, v in threshold_metrics(y, p, threshold).items()
        if k in ("tp", "fp", "fn", "tn")
    }
    ece, _ = expected_calibration_error(y, probs)
    res["calibration"] = {"ece": ece, "brier": brier_score(y, p), "nll": nll(y, probs)}
    if mc is not None:
        ece_mc, _ = expected_calibration_error(y, mc.mean(axis=1))
        res["calibration"]["ece_mc_mean"] = ece_mc

    oracle = oracle_aurc(y, pred)
    res["selective"] = {}
    for name, u in scores.items():
        a = aurc(y, pred, u)
        res["selective"][name] = {
            "aurc": a,
            "e_aurc": a - oracle,
            "auroc_error_detection": roc_auc(pred != y, u),
            **{f"acc@{int(c * 100)}%cov": accuracy_at_coverage(y, pred, u, c) for c in COVERAGES},
        }
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--checkpoints", nargs="+", required=True, help="first = reported model; all = ensemble"
    )
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--operating-point", help="JSON from an internal run; skips fitting on val")
    ap.add_argument("--target-sensitivity", type=float, default=0.95)
    ap.add_argument("--defer-rate", type=float, default=0.10, help="fraction of val cases to refer")
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--tta-samples", type=int, default=0, help="0 disables TTA")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--resize-mode", default="squash")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    device = get_device()
    manifest = read_manifest(args.manifest)
    splits = [s for s in ("val", "test") if (manifest["split"] == s).any()]
    tfm = eval_transform(args.image_size, args.resize_mode)

    runs: dict[str, list[dict]] = {s: [] for s in splits}
    tta: dict[str, np.ndarray] = {}
    for i, ckpt in enumerate(args.checkpoints):
        model = load_model(ckpt, device)
        for s in splits:
            df = manifest[manifest.split == s].reset_index(drop=True)
            ds = XrayDataset(df, args.root, tfm)
            runs[s].append(
                collect(model, ds, device, args.mc_samples if i == 0 else 0, workers=args.workers)
            )
            if i == 0 and args.tta_samples:
                tta[s] = collect_tta(
                    model,
                    df,
                    args.root,
                    device,
                    args.tta_samples,
                    args.image_size,
                    args.resize_mode,
                    args.workers,
                )
        print(f"collected predictions for {ckpt}")

    # ---- operating point: fitted on val, frozen for test / external sets
    if args.operating_point:
        op = json.loads(Path(args.operating_point).read_text())
    else:
        if "val" not in splits:
            raise SystemExit(
                "No val split in manifest: pass --operating-point from an internal run."
            )
        vdf = manifest[manifest.split == "val"]
        vy, vlog = vdf["label"].to_numpy(), runs["val"][0]["logits"]
        T = fit_temperature(vlog, vy)
        vp = softmax_np(vlog, T)
        vmc = runs["val"][0]["mc"]
        vent = MCResult(vmc, vmc.mean(1)).predictive_entropy if vmc is not None else None
        op = {
            "temperature": T,
            "threshold": select_threshold(vy, vp[:, 1], args.target_sensitivity),
            "target_sensitivity": args.target_sensitivity,
            "defer_entropy": float(np.quantile(vent, 1 - args.defer_rate))
            if vent is not None
            else None,
            "defer_rate_on_val": args.defer_rate,
            "fitted_on": str(args.manifest) + ":val",
        }
    (out / "operating_point.json").write_text(json.dumps(op, indent=2))

    metrics: dict = {"operating_point": op, "checkpoints": args.checkpoints}
    for s in splits:
        df = manifest[manifest.split == s].reset_index(drop=True)
        y = df["label"].to_numpy()
        groups = df["patient_id"].to_numpy() if "patient_id" in df else None
        primary = runs[s][0]
        probs_raw = softmax_np(primary["logits"])
        probs = softmax_np(primary["logits"], op["temperature"])
        mc = primary["mc"]
        mc_res = MCResult(mc, mc.mean(1)) if mc is not None else None
        ens = (
            np.stack([softmax_np(r["logits"]) for r in runs[s]], axis=1)
            if len(runs[s]) > 1
            else None
        )
        scores = uncertainty_scores(probs, mc_res, ens, tta.get(s))

        metrics[s] = {
            "single_model": evaluate_split(
                y, probs, mc, scores, op["threshold"], groups, args.n_boot
            )
        }
        metrics[s]["single_model"]["calibration"]["ece_before_temperature"] = (
            expected_calibration_error(y, probs_raw)[0]
        )
        if ens is not None:
            ens_mean = ens.mean(axis=1)
            metrics[s]["ensemble"] = {
                "n_models": ens.shape[1],
                "auc": with_ci(roc_auc, y, ens_mean[:, 1], groups=groups, n_boot=args.n_boot),
                **{k: v for k, v in threshold_metrics(y, ens_mean[:, 1], op["threshold"]).items()},
            }

        pred_df = df[[c for c in ("path", "label", "patient_id", "subtype") if c in df]].copy()
        pred_df["p_pneumonia"] = probs[:, 1]
        pred_df["pred"] = (probs[:, 1] >= op["threshold"]).astype(int)
        for k, v in scores.items():
            pred_df[f"u_{k}"] = v
        pred_df.to_csv(out / f"predictions_{s}.csv", index=False)

        pred = pred_df["pred"].to_numpy()
        tag = Path(args.manifest).stem + f"_{s}"
        plot_roc(
            {
                "single model": (y, probs[:, 1]),
                **({"ensemble": (y, ens.mean(1)[:, 1])} if ens is not None else {}),
            },
            out / "figures" / f"roc_{tag}",
            op=(
                metrics[s]["single_model"]["sensitivity"]["value"],
                metrics[s]["single_model"]["specificity"]["value"],
            ),
        )
        plot_reliability(
            y,
            {"uncalibrated": probs_raw, f"temperature T={op['temperature']:.2f}": probs},
            out / "figures" / f"reliability_{tag}",
        )
        plot_risk_coverage(y, pred, scores, out / "figures" / f"risk_coverage_{tag}")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(
        pd.json_normalize(metrics.get("test", metrics.get("val")), sep=".").T.head(40).to_string()
    )
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
