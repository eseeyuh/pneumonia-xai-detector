"""High-level inference used by the Streamlit demo and the REST API."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pxai import CLASS_NAMES
from pxai.model import download_weights, load_model
from pxai.preprocessing import batch_from_pil, looks_like_chest_xray, to_display_rgb
from pxai.uncertainty import mc_dropout, softmax_np
from pxai.xai import grad_cam, overlay


@dataclass
class OperatingPoint:
    """Decision settings, ideally produced by ``pxai.evaluate`` on the validation set.

    ``threshold`` is on P(pneumonia). ``defer_entropy`` is the MC predictive
    entropy above which a case is referred for human review.
    """

    threshold: float = 0.5
    defer_entropy: float = 0.35
    temperature: float = 1.0
    source: str = "default (not calibrated)"

    @classmethod
    def from_json(cls, path: str | Path) -> OperatingPoint:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            **{k: data[k] for k in ("threshold", "defer_entropy", "temperature") if k in data},
            source=str(path),
        )


@dataclass
class Prediction:
    label: str
    p_pneumonia: float
    p_pneumonia_mc: float
    predictive_entropy: float
    mutual_information: float
    variance: float
    refer_to_human: bool
    input_warnings: list[str] = field(default_factory=list)
    mc_samples: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Predictor:
    """Thread-safe wrapper: no module state is mutated at inference time."""

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str | torch.device = "cpu",
        operating_point: OperatingPoint | None = None,
        image_size: int = 224,
        resize_mode: str = "squash",
    ):
        path = Path(weights) if weights else Path(download_weights())
        self.device = torch.device(device)
        self.model = load_model(path, self.device)
        self.op = operating_point or OperatingPoint()
        self.image_size, self.resize_mode = image_size, resize_mode

    def _tensor(self, img: Image.Image) -> torch.Tensor:
        return batch_from_pil(img, self.image_size, self.resize_mode).to(self.device)

    @torch.no_grad()
    def predict(self, img: Image.Image, n_samples: int = 30) -> Prediction:
        ok, warnings = looks_like_chest_xray(img)
        x = self._tensor(img)
        logits = self.model(x).cpu().numpy()
        probs = softmax_np(logits, self.op.temperature)[0]
        mc = mc_dropout(self.model, x, n_samples=n_samples, temperature=self.op.temperature)
        p = float(probs[1])
        ent = float(mc.predictive_entropy[0])
        return Prediction(
            label=CLASS_NAMES[int(p >= self.op.threshold)],
            p_pneumonia=p,
            p_pneumonia_mc=float(mc.mean_probs[0, 1]),
            predictive_entropy=ent,
            mutual_information=float(mc.mutual_information[0]),
            variance=float(mc.variance[0]),
            refer_to_human=bool(ent >= self.op.defer_entropy or not ok),
            input_warnings=warnings,
            mc_samples=mc.samples[0, :, 1].round(4).tolist(),
        )

    def explain(self, img: Image.Image, target: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (display RGB in [0,1], Grad-CAM overlay uint8)."""
        cam = grad_cam(self.model, self._tensor(img), target)[0]
        rgb = to_display_rgb(img, self.image_size, self.resize_mode)
        return rgb, overlay(rgb, cam)
