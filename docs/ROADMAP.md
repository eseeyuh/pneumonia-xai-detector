# Roadmap

Status as of v0.2 (2026-09-29). ✅ done in code · ⬜ to do.

## Phase 1 — Reproducible pipeline ✅

- ✅ Package structure (`pxai/`), config files, CLI for split / train / evaluate
- ✅ Patient-level splits with a hard leakage check and a leakage audit
- ✅ Training with seeds, AMP, early stopping, history, optional W&B
- ✅ Evaluation: CIs (cluster bootstrap), calibration, risk–coverage, ensemble and TTA baselines
- ✅ Grad-CAM / Grad-CAM++ with localisation metrics and sanity check
- ✅ RSNA preparation script
- ✅ 35 tests, ruff, GitHub Actions CI
- ✅ Colab notebook that reproduces everything

## Phase 2 — Run the experiments ⬜

- ⬜ Download Kermany; run `--audit` and record the leakage numbers
- ⬜ Train 5 seeds on the patient-level split (Colab T4: ~15–20 min per seed)
- ⬜ Internal evaluation with `--tta-samples 16`
- ⬜ Copy `results/kermany/operating_point.json` → `configs/operating_point.json`
- ⬜ RSNA external evaluation and saliency evaluation
- ⬜ Upload the new `best.pt` to Hugging Face (keep the old file as `v0.1/best.pt`), pin the revision in `pxai/model.py`
- ⬜ Paste `docs/MODEL_CARD.md` (with real numbers) into the Hugging Face model card
- ⬜ Fill the Results table in the README

## Phase 3 — Paper ⬜

- ⬜ Write Methods straight from `docs/PAPER_PROTOCOL.md`
- ⬜ Tables 1–4, Figures 1–6
- ⬜ Error analysis: 10 most confident FN/FP with Grad-CAM
- ⬜ Paired bootstrap test: MC entropy vs MSP (AURC difference)
- ⬜ Limitations section
- ⬜ Supervisor review → arXiv → conference/journal

## Phase 4 — Portfolio polish ⬜

- ⬜ New demo screenshot/GIF after the operating point is calibrated
- ⬜ Deploy the API (e.g. Hugging Face Spaces with Docker) and add the link
- ⬜ ONNX export + latency table (CPU/GPU, batch 1)
- ⬜ Short write-up / blog post: "Auditing my own medical AI project: data leakage, calibration and what Grad-CAM really shows"
- ⬜ Pin the repo on GitHub; add topics (`medical-imaging`, `explainable-ai`, `uncertainty-quantification`, `pytorch`)
