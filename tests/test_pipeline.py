"""End-to-end smoke tests: split → train → evaluate → serve."""

import json

import numpy as np
import pytest
from PIL import Image

from pxai import data, evaluate, train
from pxai.inference import OperatingPoint, Predictor


@pytest.fixture(scope="module")
def manifest(kermany_tree, tmp_path_factory):
    out = tmp_path_factory.mktemp("splits") / "kermany.csv"
    data.main(
        [
            "--root",
            str(kermany_tree),
            "--mode",
            "patient",
            "--val-fraction",
            "0.25",
            "--test-fraction",
            "0.25",
            "--out",
            str(out),
        ]
    )
    return out


def test_train_one_epoch(kermany_tree, manifest, tmp_path):
    cfg = {
        "seed": 0,
        "data": {
            "root": str(kermany_tree),
            "manifest": str(manifest),
            "image_size": 64,
            "resize_mode": "squash",
            "hflip": False,
            "num_workers": 0,
        },
        "model": {"dropout": 0.3, "pretrained": False},
        "train": {
            "epochs": 1,
            "batch_size": 8,
            "lr": 1e-4,
            "weight_decay": 0.0,
            "warmup_epochs": 0,
            "class_weighting": True,
            "label_smoothing": 0.0,
            "amp": False,
            "early_stopping_patience": 3,
            "monitor": "val_auc",
        },
        "output_dir": str(tmp_path / "run"),
    }
    best = train.train(cfg)
    assert best.exists()
    assert (tmp_path / "run" / "history.csv").exists()


def test_evaluate_end_to_end(kermany_tree, manifest, checkpoint, tmp_path):
    out = tmp_path / "eval"
    evaluate.main(
        [
            "--manifest",
            str(manifest),
            "--root",
            str(kermany_tree),
            "--checkpoints",
            checkpoint,
            checkpoint,
            "--out",
            str(out),
            "--image-size",
            "64",
            "--mc-samples",
            "5",
            "--tta-samples",
            "2",
            "--n-boot",
            "20",
            "--workers",
            "0",
        ]
    )
    metrics = json.loads((out / "metrics.json").read_text())
    op = json.loads((out / "operating_point.json").read_text())
    assert {"threshold", "temperature", "defer_entropy"} <= set(op)
    test = metrics["test"]["single_model"]
    assert 0 <= test["auc"]["value"] <= 1
    assert {"msp", "mc_entropy", "mc_mutual_info", "ensemble_entropy", "tta_entropy"} <= set(
        test["selective"]
    )
    assert (out / "figures" / "risk_coverage_kermany_test.png").exists()
    assert (out / "predictions_test.csv").exists()

    # external evaluation reuses the frozen operating point
    ext = tmp_path / "ext"
    evaluate.main(
        [
            "--manifest",
            str(manifest),
            "--root",
            str(kermany_tree),
            "--checkpoints",
            checkpoint,
            "--out",
            str(ext),
            "--image-size",
            "64",
            "--mc-samples",
            "3",
            "--n-boot",
            "10",
            "--workers",
            "0",
            "--operating-point",
            str(out / "operating_point.json"),
        ]
    )
    assert json.loads((ext / "operating_point.json").read_text()) == op


def test_predictor(checkpoint, tmp_path):
    op_file = tmp_path / "op.json"
    op_file.write_text(json.dumps({"threshold": 0.3, "defer_entropy": 0.9, "temperature": 1.5}))
    p = Predictor(
        weights=checkpoint, operating_point=OperatingPoint.from_json(op_file), image_size=64
    )
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (80, 80), dtype=np.uint8))
    pred = p.predict(img, n_samples=10)
    assert pred.label in {"NORMAL", "PNEUMONIA"}
    assert pred.label == ("PNEUMONIA" if pred.p_pneumonia >= 0.3 else "NORMAL")
    assert len(pred.mc_samples) == 10
    rgb, ov = p.explain(img)
    assert rgb.shape == (64, 64, 3) and ov.dtype == np.uint8


def test_api(checkpoint, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import api.main as api_main

    monkeypatch.setenv("PXAI_WEIGHTS", checkpoint)
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (96, 96), dtype=np.uint8))
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    with TestClient(api_main.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post(
            "/predict?mc_samples=5", files={"file": ("x.png", buf.getvalue(), "image/png")}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert 0 <= body["p_pneumonia"] <= 1 and "refer_to_human" in body
        bad = client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")})
        assert bad.status_code == 400
