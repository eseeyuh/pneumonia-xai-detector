"""Dataset indexing and **patient-level** splitting.

The Kermany et al. (2018) paediatric chest X-ray set has several images per
child. Splitting by image lets the same child appear in train and test, which
inflates every metric. Everything here splits by patient.

CLI::

    python -m pxai.data --root data/chest_xray --mode official --out data/splits/kermany.csv
    python -m pxai.data --root data/chest_xray --audit          # measure leakage of a random split
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pxai.preprocessing import load_image

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".dcm"}
LABELS = {"NORMAL": 0, "PNEUMONIA": 1}
SPLIT_NAMES = {"train", "val", "test"}

_PNEU_RE = re.compile(r"^(person\d+)_(bacteria|virus)_\d+", re.IGNORECASE)
_NORMAL_RE = re.compile(r"^((?:NORMAL2-)?IM-\d+)-\d+", re.IGNORECASE)


def parse_kermany_filename(name: str, label: str) -> tuple[str, str | None]:
    """Return (patient_id, subtype) from a Kermany filename.

    ``person1234_bacteria_5678.jpeg`` → (``PNEUMONIA:person1234``, ``bacteria``)
    ``IM-0115-0001.jpeg``            → (``NORMAL:IM-0115``, None)
    ``NORMAL2-IM-1427-0001.jpeg``    → (``NORMAL:NORMAL2-IM-1427``, None)

    Patient IDs are namespaced by class folder because the two classes use
    unrelated numbering schemes.
    """
    stem = Path(name).stem
    m = _PNEU_RE.match(stem)
    if m:
        return f"{label}:{m.group(1).lower()}", m.group(2).lower()
    m = _NORMAL_RE.match(stem)
    if m:
        return f"{label}:{m.group(1).upper()}", None
    return f"{label}:{stem}", None  # unknown pattern: treat image as its own patient


def index_kermany(root: str | Path) -> pd.DataFrame:
    """Walk a Kermany-style tree (``<split>/<LABEL>/<image>``) into a DataFrame."""
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_SUFFIXES or p.name.startswith("."):
            continue
        parts = [s.lower() for s in p.relative_to(root).parts]
        label_name = p.parent.name.upper()
        if label_name not in LABELS:
            continue
        official = next((s for s in parts if s in SPLIT_NAMES), None)
        pid, subtype = parse_kermany_filename(p.name, label_name)
        rows.append(
            {
                "path": str(p.relative_to(root)).replace("\\", "/"),
                "label": LABELS[label_name],
                "label_name": label_name,
                "patient_id": pid,
                "subtype": subtype,
                "official_split": official,
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"No images found under {root} (expected <split>/NORMAL|PNEUMONIA/*)"
        )
    df = pd.DataFrame(rows)
    # macOS resource forks / duplicates across Kaggle mirrors
    return df.drop_duplicates(subset=["path"]).reset_index(drop=True)


def _split_patients(
    patients: pd.DataFrame, fractions: dict[str, float], rng: np.random.Generator
) -> dict[str, str]:
    """Stratified (by patient label) assignment of patient IDs to splits."""
    assert abs(sum(fractions.values()) - 1.0) < 1e-6, fractions
    out: dict[str, str] = {}
    for _, grp in patients.groupby("label"):
        ids = grp["patient_id"].to_numpy().copy()
        rng.shuffle(ids)
        cuts = np.cumsum([int(round(f * len(ids))) for f in fractions.values()])[:-1]
        for name, chunk in zip(fractions, np.split(ids, cuts), strict=True):
            out.update({pid: name for pid in chunk})
    return out


def make_splits(
    df: pd.DataFrame,
    mode: str = "official",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign ``split`` ∈ {train, val, test} at the patient level.

    ``official``: keep Kermany's official test set; re-split the rest into
    train/val by patient (the official 16-image val set is too small to use).
    ``patient``: pool everything and split by patient.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    patients = df.groupby("patient_id", as_index=False)["label"].first()

    if mode == "official":
        if df["official_split"].isna().any():
            raise ValueError("mode='official' needs the train/val/test folder layout.")
        is_test = df["official_split"] == "test"
        dev_patients = patients[patients["patient_id"].isin(df.loc[~is_test, "patient_id"])]
        # a patient that also appears in the official test set stays out of train/val
        dev_patients = dev_patients[~dev_patients["patient_id"].isin(df.loc[is_test, "patient_id"])]
        assign = _split_patients(
            dev_patients, {"train": 1 - val_fraction, "val": val_fraction}, rng
        )
        df["split"] = np.where(is_test, "test", df["patient_id"].map(assign))
        df = df[df["split"].notna()].reset_index(drop=True)
    elif mode == "patient":
        fr = {"train": 1 - val_fraction - test_fraction, "val": val_fraction, "test": test_fraction}
        df["split"] = df["patient_id"].map(_split_patients(patients, fr, rng))
    else:
        raise ValueError(f"Unknown split mode {mode!r}")
    return df


def patient_overlap(df: pd.DataFrame) -> dict[str, int]:
    """Number of patients shared between each pair of splits (should all be 0)."""
    sets = {s: set(df.loc[df["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
    return {
        "train&val": len(sets["train"] & sets["val"]),
        "train&test": len(sets["train"] & sets["test"]),
        "val&test": len(sets["val"] & sets["test"]),
    }


def audit_random_split(df: pd.DataFrame, test_fraction: float = 0.1, seed: int = 0) -> dict:
    """Quantify leakage of a naive *image-level* random split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(round(test_fraction * len(df)))
    test, train = df.iloc[idx[:n_test]], df.iloc[idx[n_test:]]
    leaked = test["patient_id"].isin(set(train["patient_id"]))
    return {
        "n_images": len(df),
        "n_patients": df["patient_id"].nunique(),
        "images_per_patient_mean": float(df.groupby("patient_id").size().mean()),
        "test_images": n_test,
        "test_images_with_patient_in_train": int(leaked.sum()),
        "leak_fraction": float(leaked.mean()),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "label_name"])
        .agg(images=("path", "size"), patients=("patient_id", "nunique"))
        .unstack("label_name")
    )


class XrayDataset(Dataset):
    """Reads rows of a split manifest (``path``, ``label``, optionally ``patient_id``)."""

    def __init__(self, manifest: pd.DataFrame, root: str | Path, transform):
        self.df = manifest.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def load_pil(self, i: int):
        p = Path(self.df.at[i, "path"])
        return load_image(p if p.is_absolute() else self.root / p)

    def __getitem__(self, i: int):
        return self.transform(self.load_pil(i)), torch.tensor(int(self.df.at[i, "label"])), i


def read_manifest(path: str | Path, split: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", required=True, help="Folder with train/val/test/NORMAL|PNEUMONIA")
    ap.add_argument("--mode", default="official", choices=["official", "patient"])
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/splits/kermany.csv")
    ap.add_argument("--audit", action="store_true", help="Only report leakage of a random split")
    args = ap.parse_args(argv)

    df = index_kermany(args.root)
    if args.audit:
        for k, v in audit_random_split(df).items():
            print(f"{k:>36}: {v}")
        return

    split = make_splits(df, args.mode, args.val_fraction, args.test_fraction, args.seed)
    overlap = patient_overlap(split)
    if any(overlap.values()):
        raise RuntimeError(f"Patient leakage detected: {overlap}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.out, index=False)
    print(summarize(split).to_string())
    print(f"\nPatient overlap between splits: {overlap}")
    if dropped := len(df) - len(split):
        print(
            f"Dropped {dropped} train/val images of patients who are also in the official test set"
        )
    print(f"Wrote {len(split)} rows → {args.out}")


if __name__ == "__main__":
    main()
