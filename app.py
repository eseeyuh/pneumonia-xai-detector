"""Streamlit demo: classification, Grad-CAM and MC-Dropout uncertainty.

    streamlit run app.py

Research prototype — not a medical device.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import torch

from pxai.inference import OperatingPoint, Predictor
from pxai.preprocessing import load_image

WEIGHTS = os.environ.get("PXAI_WEIGHTS")  # local path; if unset, pulled from the HF Hub
OP_PATH = Path(os.environ.get("PXAI_OPERATING_POINT", "configs/operating_point.json"))

C_INK, C_NORMAL, C_PNEU, C_AMBER, C_MUTED = "#16202b", "#1f9d8b", "#d1495b", "#e0a458", "#5c6b7a"


@st.cache_resource(show_spinner="Loading model weights…")
def get_predictor() -> Predictor:
    op = OperatingPoint.from_json(OP_PATH) if OP_PATH.exists() else OperatingPoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Predictor(weights=WEIGHTS, device=device, operating_point=op)


def card(label: str, value: str, color: str = C_INK, sub: str = "") -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color}">{value}</div>{sub_html}</div>'
    )


def uncertainty_plot(samples: list[float], threshold: float):
    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=120)
    ax.hist(samples, bins=20, range=(0, 1), color=C_PNEU, alpha=0.75, edgecolor="white")
    ax.axvline(threshold, color=C_MUTED, ls="--", lw=1, label=f"decision threshold {threshold:.2f}")
    mean = sum(samples) / len(samples)
    ax.axvline(mean, color=C_INK, lw=2, label=f"mean {mean:.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(pneumonia) per stochastic pass", fontsize=9)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="XAI Pneumonia Triage", layout="wide")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
.stApp { background: #f4f6f8; }
html, body, .stMarkdown, p, label { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1150px; }
.app-title { font-size: 2rem; font-weight: 700; color: #16202b; letter-spacing: -0.02em; }
.app-sub { color: #5c6b7a; font-size: .98rem; margin-bottom: 1rem; }
.metric-card { background: #fff; border: 1px solid #e3e8ec; border-radius: 14px; padding: 18px 20px; height: 100%; }
.metric-label { font-size: .74rem; letter-spacing: .08em; text-transform: uppercase; color: #5c6b7a; font-weight: 600; margin-bottom: 6px; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 600; line-height: 1.1; }
.metric-sub { font-size: .82rem; color: #5c6b7a; margin-top: 4px; }
.verdict { border-radius: 14px; padding: 22px 26px; color: #fff; }
.verdict-label { font-size: .8rem; letter-spacing: .1em; text-transform: uppercase; opacity: .85; font-weight: 600; }
.verdict-class { font-size: 2.2rem; font-weight: 700; }
.section-h { font-size: 1.15rem; font-weight: 600; color: #16202b; margin: 1.6rem 0 .3rem 0; }
.disclaimer { background: #fff6e9; border: 1px solid #f0d9b5; border-radius: 12px; padding: 12px 16px; font-size: .86rem; color: #7a5a23; margin: .4rem 0 1.2rem 0; }
[data-testid="stImage"] img { border-radius: 12px; border: 1px solid #e3e8ec; }
</style>""",
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Explainable Pneumonia Triage</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">DenseNet-121 · Grad-CAM · MC-Dropout uncertainty with '
    "selective referral</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="disclaimer"><b>Research prototype, not a medical device.</b> The model was trained '
    "on paediatric (age 1–5) chest X-rays from a single hospital (Kermany et al., 2018). It has not "
    "been validated for adults, other scanners or clinical use. Do not upload identifiable patient "
    "data.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Settings")
    n_samples = st.slider(
        "MC-Dropout samples (T)",
        10,
        100,
        30,
        step=10,
        help="Only the classifier head is resampled, so T is cheap.",
    )
    show_cam = st.checkbox("Grad-CAM heatmap", value=True)
    st.markdown("---")

try:
    predictor = get_predictor()
except Exception as e:  # noqa: BLE001 - show any load failure to the user
    st.error(f"Could not load the model: {e}")
    st.stop()

with st.sidebar:
    st.caption(f"Device: **{predictor.device.type.upper()}**")
    st.caption(f"Decision threshold: **{predictor.op.threshold:.2f}**")
    st.caption(f"Refer if entropy ≥ **{predictor.op.defer_entropy:.2f}**")
    st.caption(f"Operating point: {predictor.op.source}")

uploaded = st.file_uploader(
    "Upload a chest X-ray (JPEG, PNG or DICOM)", type=["jpg", "jpeg", "png", "dcm", "dicom"]
)
if not uploaded:
    st.info(
        "Upload a frontal chest X-ray to get a prediction, a Grad-CAM heatmap and an "
        "uncertainty estimate."
    )
    st.stop()

try:
    img = load_image(uploaded.getvalue(), filename=uploaded.name)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the image: {e}")
    st.stop()

with st.spinner("Running model…"):
    pred = predictor.predict(img, n_samples=n_samples)
    rgb, cam_overlay = predictor.explain(img, target=1) if show_cam else (None, None)

for w in pred.input_warnings:
    st.warning(f"{w} The result below is probably meaningless.")

is_pneu = pred.label == "PNEUMONIA"
color = C_PNEU if is_pneu else C_NORMAL
referral = (
    "Refer for radiologist review — the model is uncertain"
    if pred.refer_to_human
    else "Confident prediction"
)
st.markdown(
    f'<div class="verdict" style="background:{color}"><div class="verdict-label">Model output</div>'
    f'<div class="verdict-class">{pred.label}</div><div style="opacity:.92">{referral}</div></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-h">Prediction</div>', unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
m1.markdown(
    card(
        "P(pneumonia)",
        f"{pred.p_pneumonia * 100:.1f}%",
        color,
        sub=f"threshold {predictor.op.threshold:.2f}",
    ),
    unsafe_allow_html=True,
)
m2.markdown(
    card(
        "MC mean P(pneumonia)",
        f"{pred.p_pneumonia_mc * 100:.1f}%",
        C_INK,
        sub=f"over {n_samples} samples",
    ),
    unsafe_allow_html=True,
)
ent_color = C_PNEU if pred.refer_to_human else C_NORMAL
m3.markdown(
    card(
        "Predictive entropy",
        f"{pred.predictive_entropy:.3f}",
        ent_color,
        sub="0 = certain · 0.693 = maximal",
    ),
    unsafe_allow_html=True,
)

if show_cam:
    st.markdown(
        '<div class="section-h">Where the model looked (evidence for pneumonia)</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.image((rgb * 255).astype("uint8"), caption="Model input (224×224)", width="stretch")
    c2.image(cam_overlay, caption="Grad-CAM for the PNEUMONIA class", width="stretch")
    st.caption(
        "Grad-CAM shows which regions raised the pneumonia score. Heat on borders, text "
        "markers or outside the lungs suggests the model relies on shortcuts."
    )

st.markdown('<div class="section-h">Uncertainty</div>', unsafe_allow_html=True)
u1, u2 = st.columns(2)
u1.markdown(
    card("Mutual information (epistemic)", f"{pred.mutual_information:.3f}", C_INK),
    unsafe_allow_html=True,
)
u2.markdown(card("Variance of P(pneumonia)", f"{pred.variance:.4f}", C_INK), unsafe_allow_html=True)
st.pyplot(uncertainty_plot(pred.mc_samples, predictor.op.threshold), width="stretch")
st.caption(
    "Dropout stays active in the classifier head, so each pass samples a slightly different "
    "model. A wide spread means the model is unsure and the case should be read by a human."
)
