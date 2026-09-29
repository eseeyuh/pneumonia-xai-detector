"""REST API for the pneumonia triage model.

uvicorn api.main:app --host 0.0.0.0 --port 8000
curl -F "file=@xray.png" http://localhost:8000/predict
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from pxai import __version__
from pxai.inference import OperatingPoint, Predictor
from pxai.preprocessing import load_image

MAX_BYTES = 20 * 1024 * 1024
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    op_path = Path(os.environ.get("PXAI_OPERATING_POINT", "configs/operating_point.json"))
    op = OperatingPoint.from_json(op_path) if op_path.exists() else OperatingPoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["predictor"] = Predictor(
        weights=os.environ.get("PXAI_WEIGHTS"), device=device, operating_point=op
    )
    yield
    state.clear()


app = FastAPI(
    title="Pneumonia XAI triage",
    version=__version__,
    lifespan=lifespan,
    description="Research prototype. Not a medical device.",
)


@app.get("/health")
def health() -> dict:
    p = state.get("predictor")
    return {
        "status": "ok" if p else "loading",
        "version": __version__,
        "device": str(p.device) if p else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), mc_samples: int = Query(30, ge=1, le=200)) -> dict:
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 20 MB)")
    try:
        img = load_image(data, filename=file.filename)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not decode image: {e}") from e
    pred = state["predictor"].predict(img, n_samples=mc_samples)
    out = pred.to_dict()
    out.pop("mc_samples")
    out["operating_point"] = state["predictor"].op.__dict__
    return out
