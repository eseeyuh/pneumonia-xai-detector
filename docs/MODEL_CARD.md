---
license: mit
library_name: pytorch
pipeline_tag: image-classification
tags:
  - medical-imaging
  - chest-x-ray
  - pneumonia
  - explainable-ai
  - uncertainty-quantification
datasets:
  - kermany-chest-xray
---

# Pneumonia triage — DenseNet-121 (research prototype)

Paste this card into the Hugging Face model page once the v0.2 results are computed. Replace every `TBD`.

## Model

- Architecture: DenseNet-121 (ImageNet init), head `Dropout(0.3) → Linear(1024, 2)`; classes `[NORMAL, PNEUMONIA]`
- Input: grayscale chest X-ray replicated to 3 channels, 224×224, ImageNet normalisation
- File: `best.pt`, a plain PyTorch `state_dict` (load with `pxai.model.load_model`)
- Code: https://github.com/eseeyuh/pneumonia-xai-detector

```python
from pxai.inference import Predictor
from pxai.preprocessing import load_image

pred = Predictor().predict(load_image("xray.png"))
print(pred.label, pred.p_pneumonia, pred.refer_to_human)
```

## Intended use

Research and education on explainability and uncertainty in medical imaging. **Not a medical device; not for diagnosis.**

## Training data

Kermany et al. (2018) paediatric chest X-rays (Guangzhou Women and Children's Medical Center, age 1–5), split by patient. Train/val/test sizes: TBD.

## Evaluation (95% patient-level bootstrap CIs)

| Test set | AUC | Sensitivity | Specificity | ECE |
|---|---|---|---|---|
| Kermany official test | TBD | TBD | TBD | TBD |
| RSNA (adult, external) | TBD | TBD | TBD | TBD |

Operating point (fitted on validation): threshold TBD, temperature TBD, referral entropy TBD.

## Limitations

Single-hospital paediatric data; image-level labels; possible shortcut features; uncertainty is not an out-of-distribution detector. Performance on adults is lower (see RSNA row).

## Versions

- v0.2 — retrained on patient-level split (this card)
- v0.1 — original weights; reported metrics withdrawn because the split could not be verified as patient-level
