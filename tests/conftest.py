import numpy as np
import pytest
import torch
from PIL import Image

from pxai.model import PneumoniaNet


@pytest.fixture(scope="session")
def tiny_model() -> PneumoniaNet:
    torch.manual_seed(0)
    return PneumoniaNet(pretrained=False).eval()


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory, tiny_model) -> str:
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    torch.save(tiny_model.state_dict(), path)
    return str(path)


def _xray_like(rng: np.random.Generator, size: int = 96) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size] / size
    lungs = np.exp(-((xx - 0.3) ** 2 + (yy - 0.5) ** 2) / 0.03) + np.exp(
        -((xx - 0.7) ** 2 + (yy - 0.5) ** 2) / 0.03
    )
    arr = 40 + 150 * lungs + rng.normal(0, 12, (size, size))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")


@pytest.fixture(scope="session")
def kermany_tree(tmp_path_factory):
    """Synthetic Kermany-style tree with several images per patient and one
    patient deliberately present in both official train and test."""
    root = tmp_path_factory.mktemp("chest_xray")
    rng = np.random.default_rng(0)
    layout = {
        ("train", "NORMAL"): [f"IM-{p:04d}-{i:04d}" for p in range(1, 13) for i in range(1, 3)],
        ("train", "PNEUMONIA"): [
            f"person{p}_bacteria_{p * 10 + i}" for p in range(1, 21) for i in range(2)
        ],
        ("test", "NORMAL"): [f"NORMAL2-IM-{p:04d}-0001" for p in range(1, 6)],
        ("test", "PNEUMONIA"): [f"person{p}_virus_{p}" for p in range(100, 106)]
        + ["person1_virus_999"],
    }
    for (split, label), names in layout.items():
        d = root / split / label
        d.mkdir(parents=True)
        for n in names:
            _xray_like(rng).save(d / f"{n}.jpeg")
    return root
