import io

import numpy as np
import torch
from PIL import Image

from pxai.preprocessing import batch_from_pil, load_image, looks_like_chest_xray
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


def test_grad_cam_shapes_and_range(tiny_model):
    x = torch.randn(2, 3, 64, 64)
    for fn in (grad_cam, grad_cam_pp):
        cam = fn(tiny_model, x, target=1)
        assert cam.shape == (2, 64, 64)
        assert cam.min() >= 0 and cam.max() <= 1 + 1e-6
    assert not any(p.grad is not None for p in tiny_model.parameters())


def test_overlay_dtype():
    out = overlay(np.zeros((8, 8, 3)), np.ones((8, 8)))
    assert out.dtype == np.uint8 and out.shape == (8, 8, 3)


def test_localisation_metrics():
    mask = boxes_to_mask([(10, 10, 20, 20)], (64, 64))
    cam = np.zeros((64, 64))
    cam[15:25, 15:25] = 1.0
    assert pointing_game(cam, mask)
    assert energy_in_mask(cam, mask) == 1.0
    assert 0 < iou_at_threshold(cam, mask) < 1
    cam_out = np.zeros((64, 64))
    cam_out[50, 50] = 1
    assert not pointing_game(cam_out, mask)
    assert pointing_game(cam_out, mask, tolerance=25)


def test_randomisation_changes_model(tiny_model):
    rand = randomize_head_and_last_block(tiny_model)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert not torch.allclose(rand(x), tiny_model(x))
    a = np.random.default_rng(0).random((16, 16))
    assert rank_correlation(a, a) == 1.0


def test_load_16bit_png_and_rgb_flag():
    arr = (np.linspace(0, 4095, 128 * 128).reshape(128, 128)).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    img = load_image(buf.getvalue(), filename="x.png")
    assert img.mode == "L" and np.asarray(img).max() > 200
    assert batch_from_pil(img, 64).shape == (1, 3, 64, 64)

    colour = Image.new("RGB", (100, 100), (200, 30, 30))
    ok, warnings = looks_like_chest_xray(colour)
    assert not ok and warnings
