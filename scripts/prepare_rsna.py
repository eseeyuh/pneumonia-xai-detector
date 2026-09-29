"""Prepare the RSNA Pneumonia Detection Challenge set for external validation.

Download from Kaggle (``rsna-pneumonia-detection-challenge``) and run::

    python scripts/prepare_rsna.py --rsna-dir data/rsna --out-dir data/rsna_png

Produces:
  data/rsna_png/images/<patientId>.png   (8-bit, resized to --size)
  data/splits/rsna.csv                   (path,label,patient_id,split=test,rsna_class)
  data/splits/rsna_boxes.csv             (patient_id,x,y,w,h in the resized frame)

Labels: ``Lung Opacity`` → 1, ``Normal`` → 0. ``No Lung Opacity / Not Normal``
is excluded by default (it is neither class in our binary task); pass
``--include-not-normal`` to count it as 0 for a harder negative set.

Note the population shift: RSNA is adult (NIH ChestX-ray14), Kermany is
paediatric (age 1–5). Report this as a limitation, not a bug.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from pxai.preprocessing import load_image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--rsna-dir", required=True)
    ap.add_argument("--out-dir", default="data/rsna_png")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--include-not-normal", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N images")
    args = ap.parse_args()

    rsna, out = Path(args.rsna_dir), Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(rsna / "stage_2_train_labels.csv")
    classes = pd.read_csv(rsna / "stage_2_detailed_class_info.csv").drop_duplicates("patientId")
    per_patient = classes.set_index("patientId")["class"]

    keep = {"Lung Opacity": 1, "Normal": 0}
    if args.include_not_normal:
        keep["No Lung Opacity / Not Normal"] = 0
    patients = [p for p, c in per_patient.items() if c in keep]
    if args.limit:
        patients = patients[: args.limit]

    rows, boxes = [], []
    scale = args.size / 1024.0  # RSNA images are 1024×1024
    for i, pid in enumerate(patients):
        dst = out / "images" / f"{pid}.png"
        if not dst.exists():
            img = load_image(rsna / "stage_2_train_images" / f"{pid}.dcm")
            img.resize((args.size, args.size), Image.BILINEAR).save(dst)
        rows.append(
            {
                "path": str(dst.resolve()),
                "label": keep[per_patient[pid]],
                "patient_id": pid,
                "split": "test",
                "rsna_class": per_patient[pid],
            }
        )
        for _, b in labels[(labels.patientId == pid) & (labels.Target == 1)].iterrows():
            boxes.append(
                {
                    "patient_id": pid,
                    "x": b.x * scale,
                    "y": b.y * scale,
                    "w": b.width * scale,
                    "h": b.height * scale,
                }
            )
        if i % 1000 == 0:
            print(f"{i}/{len(patients)}")

    sp = Path(args.splits_dir)
    sp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(sp / "rsna.csv", index=False)
    pd.DataFrame(boxes).to_csv(sp / "rsna_boxes.csv", index=False)
    print(pd.DataFrame(rows)["rsna_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
