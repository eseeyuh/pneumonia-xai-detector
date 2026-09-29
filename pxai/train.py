"""Train DenseNet-121 on a patient-level split.

    python -m pxai.train --config configs/default.yaml
    python -m pxai.train --config configs/default.yaml --set seed=1 output_dir=runs/seed1

Outputs in ``output_dir``: ``best.pt`` (bare state dict, drop-in for the app),
``last.pt`` (full checkpoint), ``history.csv``, ``config.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pxai.data import XrayDataset, patient_overlap, read_manifest
from pxai.metrics import roc_auc, threshold_metrics
from pxai.model import build_model
from pxai.preprocessing import eval_transform, train_transform
from pxai.utils import get_device, load_config, save_yaml, seed_everything


def make_loaders(cfg: dict):
    d = cfg["data"]
    manifest = read_manifest(d["manifest"])
    overlap = patient_overlap(manifest)
    if any(overlap.values()):
        raise RuntimeError(f"Refusing to train: patients shared between splits {overlap}")
    size, mode = d["image_size"], d["resize_mode"]
    tr = XrayDataset(
        manifest[manifest.split == "train"], d["root"], train_transform(size, mode, d["hflip"])
    )
    va = XrayDataset(manifest[manifest.split == "val"], d["root"], eval_transform(size, mode))
    kw = dict(
        num_workers=d["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=d["num_workers"] > 0,
    )
    bs = cfg["train"]["batch_size"]
    return (
        DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True, **kw),
        DataLoader(va, batch_size=bs * 2, shuffle=False, **kw),
        tr.df["label"].to_numpy(),
    )


@torch.no_grad()
def predict_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for x, y, _ in loader:
        logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train(cfg: dict) -> Path:
    seed_everything(cfg["seed"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out / "config.yaml")
    device = get_device()
    tc = cfg["train"]

    train_loader, val_loader, train_labels = make_loaders(cfg)
    model = build_model(dropout=cfg["model"]["dropout"], pretrained=cfg["model"]["pretrained"]).to(
        device
    )

    weight = None
    if tc["class_weighting"]:
        counts = np.bincount(train_labels, minlength=2).astype(np.float32)
        weight = torch.tensor(counts.sum() / (2 * counts), device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=tc["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps = len(train_loader) * tc["epochs"]
    scheduler = cosine_with_warmup(optimizer, len(train_loader) * tc["warmup_epochs"], steps)
    use_amp = tc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    wb = None
    if cfg.get("wandb", {}).get("enabled"):
        import wandb

        wb = wandb.init(project=cfg["wandb"]["project"], config=cfg, dir=str(out))

    history, best, patience = [], -np.inf, 0
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(y)
            n += len(y)

        logits, labels = predict_loader(model, val_loader, device)
        p = torch.softmax(torch.from_numpy(logits), 1)[:, 1].numpy()
        val_loss = float(
            nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels))
        )
        m = threshold_metrics(labels, p)
        row = {
            "epoch": epoch,
            "train_loss": running / n,
            "val_loss": val_loss,
            "val_auc": roc_auc(labels, p),
            "val_balanced_accuracy": m["balanced_accuracy"],
            "val_sensitivity": m["sensitivity"],
            "val_specificity": m["specificity"],
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            " | ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
            )
        )
        if wb:
            wb.log(row)

        score = row[tc["monitor"]] * (-1 if tc["monitor"].endswith("loss") else 1)
        if score > best:
            best, patience = score, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            patience += 1
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": cfg,
            },
            out / "last.pt",
        )
        if patience >= tc["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    if wb:
        wb.finish()
    print(f"Best {tc['monitor']}: {abs(best):.4f} → {out / 'best.pt'}")
    return out / "best.pt"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = ap.parse_args(argv)
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
