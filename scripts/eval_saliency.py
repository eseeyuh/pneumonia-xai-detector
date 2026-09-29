"""Quantitative Grad-CAM evaluation against radiologist boxes (RSNA).

    python scripts/eval_saliency.py --manifest data/splits/rsna.csv \\
        --boxes data/splits/rsna_boxes.csv --checkpoint runs/seed42/best.pt --out results/saliency

For every positive image with boxes it reports, for Grad-CAM and Grad-CAM++:
  * pointing game hit rate (max of the map inside a box)
  * energy-based pointing game (share of map mass inside boxes)
  * IoU of the map thresholded at 0.5
and a baseline that "explains" with a centred Gaussian blob, so the numbers
have a floor to beat. It also runs the model-randomisation sanity check
(Adebayo et al., 2018): Spearman correlation between maps of the trained and a
partially randomised model should be low.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pxai.model import load_model
from pxai.preprocessing import eval_transform, load_image, to_display_rgb
from pxai.utils import get_device
from pxai.xai import (
    boxes_to_mask,
    energy_in_mask,
    grad_cam,
    grad_cam_pp,
    iou_at_threshold,
    overlay,
    pointing_game,
    randomize_head_and_last_block,
    rank_correlation,
)


def center_prior(size: int, sigma: float = 0.25) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1) - 0.5
    return np.exp(-(xx**2 + yy**2) / (2 * sigma**2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--source-size", type=int, default=512, help="size the boxes are expressed in")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-examples", type=int, default=12)
    ap.add_argument("--out", default="results/saliency")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    device = get_device()
    model = load_model(args.checkpoint, device)
    rand_model = randomize_head_and_last_block(model)
    tfm = eval_transform(args.image_size)
    df = pd.read_csv(args.manifest)
    boxes = pd.read_csv(args.boxes)
    df = df[(df.label == 1) & df.patient_id.isin(boxes.patient_id)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    s = args.image_size / args.source_size
    prior = center_prior(args.image_size)

    rows = []
    for i, r in df.iterrows():
        p = Path(r.path)
        img = load_image(p if p.is_absolute() else Path(args.root) / p)
        x = tfm(img).unsqueeze(0).to(device)
        b = boxes[boxes.patient_id == r.patient_id]
        mask = boxes_to_mask(
            [(bx.x * s, bx.y * s, bx.w * s, bx.h * s) for bx in b.itertuples()],
            (args.image_size, args.image_size),
        )
        maps = {
            "gradcam": grad_cam(model, x, 1)[0],
            "gradcam_pp": grad_cam_pp(model, x, 1)[0],
            "center_prior": prior,
        }
        row = {
            "patient_id": r.patient_id,
            "p_pneumonia": float(torch.softmax(model(x), 1)[0, 1].detach()),
        }
        for name, cam in maps.items():
            row[f"{name}_pointing"] = pointing_game(cam, mask)
            row[f"{name}_energy"] = energy_in_mask(cam, mask)
            row[f"{name}_iou"] = iou_at_threshold(cam, mask)
        row["mask_area_fraction"] = float(mask.mean())
        row["sanity_spearman"] = rank_correlation(maps["gradcam"], grad_cam(rand_model, x, 1)[0])
        rows.append(row)

        if i < args.n_examples:
            import matplotlib.pyplot as plt

            rgb = to_display_rgb(img, args.image_size)
            fig, ax = plt.subplots(1, 2, figsize=(6, 3))
            ax[0].imshow(rgb)
            ax[1].imshow(overlay(rgb, maps["gradcam"]))
            for a in ax:
                for bx in b.itertuples():
                    a.add_patch(
                        plt.Rectangle(
                            (bx.x * s, bx.y * s), bx.w * s, bx.h * s, fill=False, ec="lime", lw=1.2
                        )
                    )
                a.axis("off")
            fig.suptitle(f"P(pneumonia)={row['p_pneumonia']:.2f}", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / "examples" / f"{r.patient_id}.png", dpi=150)
            plt.close(fig)

    res = pd.DataFrame(rows)
    res.to_csv(out / "saliency_per_image.csv", index=False)
    summary = {c: float(res[c].mean()) for c in res.columns if c not in ("patient_id",)}
    summary["n_images"] = len(res)
    (out / "saliency_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
