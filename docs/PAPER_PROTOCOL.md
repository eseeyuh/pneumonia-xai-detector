# Paper protocol

Working title: **Explainable Pneumonia Triage from Chest X-Rays: Uncertainty-Based Referral and Quantitative Evaluation of Grad-CAM**

This document fixes the experimental protocol *before* the numbers are known, so the paper cannot be tuned to the test set. Change it only with a dated note at the bottom.

## Research questions

- **RQ1 – Discrimination.** How well does a DenseNet-121 separate pneumonia from normal on a leak-free, patient-level split, and how much does performance drop on an external adult dataset?
- **RQ2 – Selective prediction.** Does MC-Dropout uncertainty identify errors better than cheap baselines (max softmax probability, softmax entropy), TTA and a 5-model deep ensemble? How many cases must be referred to reach a target accuracy?
- **RQ3 – Explanation faithfulness.** Do Grad-CAM maps point at radiologist-marked opacities more often than a trivial centre prior, and do they pass the model-randomisation sanity check?

## Data

| Set | Role | Unit of split | Notes |
|---|---|---|---|
| Kermany et al. 2018, official `train`+`val` | train / validation (85/15 by patient) | patient | patient IDs parsed from filenames (`personN_…`, `IM-NNNN-…`) |
| Kermany official `test` (624 images) | internal test | – | kept intact for comparability with prior work; patients shared with train are removed from train |
| RSNA Pneumonia Detection Challenge | external test | – | adults; `Lung Opacity` = 1, `Normal` = 0, `Not Normal` excluded (sensitivity analysis: included as 0) |

Report in *Methods*: number of images and patients per split and class, the output of `python -m pxai.data --audit` (share of test images whose patient is in train under a naive image-level split).

## Model and training

DenseNet-121, ImageNet initialisation, head `Dropout(0.3) → Linear(1024, 2)`, 224×224 input, AdamW (lr 1e-4, wd 1e-4), cosine schedule with 1 warm-up epoch, class-weighted cross-entropy, mild affine and intensity augmentation (no horizontal flip), early stopping on validation AUC (patience 6, max 25 epochs). **Five seeds** (42, 1, 2, 3, 4). Seed 42 is the reported single model; all five form the ensemble. Report mean ± SD across seeds for the headline metrics as well.

## Decisions fixed on validation, frozen for every test set

1. **Temperature** T by NLL minimisation (temperature scaling).
2. **Decision threshold**: highest threshold with validation sensitivity ≥ 0.95 (triage: a missed pneumonia costs more than a false alarm).
3. **Referral cut-off**: MC predictive entropy at the 90th percentile of validation (≈10% referral).

## Metrics

- Discrimination: ROC-AUC, average precision; sensitivity, specificity, PPV, NPV, F1, balanced accuracy at the frozen threshold.
- Uncertainty: 95% CIs by **patient-level cluster bootstrap** (2000 resamples).
- Calibration: ECE (15 bins), Brier, NLL, before and after temperature scaling; reliability diagram.
- Selective prediction: risk–coverage curve, AURC, E-AURC, AUROC of error detection, accuracy at 70/80/90% coverage. Compare: MSP, softmax entropy, MC entropy, MC mutual information, MC variance, TTA entropy, ensemble entropy, ensemble mutual information.
- Saliency (RSNA positives with boxes): pointing-game hit rate, energy inside boxes, IoU at 0.5; Grad-CAM vs Grad-CAM++ vs centre-prior baseline; Spearman correlation after randomising the head and the last dense block.
- Statistical comparison of uncertainty scores: paired bootstrap of the AURC difference (MC entropy − MSP).

## Planned tables and figures

| # | Content | Source |
|---|---|---|
| Table 1 | Dataset statistics per split (images, patients, prevalence) | `pxai.data` output |
| Table 2 | Internal and external performance with 95% CIs, single model and ensemble | `results/*/metrics.json` |
| Table 3 | Selective prediction: AURC / E-AURC / acc@coverage for every uncertainty score | `metrics.json → selective` |
| Table 4 | Saliency localisation and sanity check | `results/saliency/saliency_summary.json` |
| Fig 1 | Pipeline overview | draw from README diagram |
| Fig 2 | ROC internal vs external with operating point | `figures/roc_*` |
| Fig 3 | Reliability diagrams before/after temperature scaling | `figures/reliability_*` |
| Fig 4 | Risk–coverage curves | `figures/risk_coverage_*` |
| Fig 5 | Grad-CAM examples with RSNA boxes: hits, misses, shortcut cases | `results/saliency/examples/` |
| Fig 6 | Error gallery: most confident false negatives and false positives | `predictions_test.csv` |

## Reporting checklist (CLAIM 2024, abbreviated)

| Item | Where it is addressed |
|---|---|
| Study design, intended use (triage support, not diagnosis) | Introduction, README "Limitations" |
| Data sources, inclusion/exclusion, demographics | Data table above; RSNA class mapping |
| Patient-level partitioning, no leakage | `pxai.data`, `patient_overlap` check, audit numbers |
| Reference standard and its limitations | Kermany image-level labels; RSNA radiologist boxes |
| Preprocessing, augmentation | `pxai/preprocessing.py`, config |
| Model, initialisation, training details, seeds | `configs/default.yaml`, Methods |
| Operating point chosen on validation only | "Decisions fixed on validation" |
| Metrics with CIs, calibration | `pxai/evaluate.py` |
| External validation | RSNA |
| Explainability evaluated, not only shown | `scripts/eval_saliency.py` |
| Failure analysis | Fig 6 |
| Code and weights availability | GitHub (MIT), Hugging Face |

Also check TRIPOD+AI if submitting to a clinical journal.

## Honest framing

- If MC-Dropout does **not** beat MSP or softmax entropy, report it. With dropout only in the head this is a likely outcome and a useful one: it tells practitioners the cheap baseline is enough.
- If performance drops sharply on RSNA, report the drop and discuss the paediatric → adult shift; do not remove the experiment.
- Never compare against numbers from papers that used a different split as if they were comparable.

## Candidate venues (check current deadlines)

- **MIUA** (Medical Image Understanding and Analysis, UK conference; good fit and visible to UK employers)
- **UNSURE** workshop at MICCAI (uncertainty in medical imaging)
- **iMIMIC** workshop at MICCAI (interpretability in medical imaging)
- Journals: *Computers in Biology and Medicine*, *IEEE Journal of Biomedical and Health Informatics*, *Scientific Reports*
- Put the preprint on arXiv (cs.CV / eess.IV) once Table 2 is final.

## Change log

- 2026-09-29: protocol written; v0.1 results withdrawn (split not verifiable as patient-level).
