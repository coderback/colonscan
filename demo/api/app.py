"""CPU inference for the ColonoScan public demo.

Runs the repo's trained EfficientNet-b4 UNet++ on a fixed list of Kvasir-SEG
images. There is deliberately no upload endpoint, so the service can only ever
process public-dataset images.
"""
import base64
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "efficient_unetpp.pth"
SAMPLES = ROOT / "samples"
IMG_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "https://colonoscan-demo.vercel.app").split(",")
    if o.strip()
]

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

app = FastAPI(title="ColonoScan demo API", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_model = None
_lock = threading.Lock()
_manifest = json.loads((SAMPLES / "manifest.json").read_text())
_ids = {s["id"] for s in _manifest}


def get_model():
    global _model
    if _model is None:
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=1
        )
        model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
        model.eval()
        _model = model
    return _model


def segment(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Same preprocessing as the notebook's validation transform."""
    resized = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0)
    started = time.perf_counter()
    with _lock, torch.inference_mode():
        probs = torch.sigmoid(get_model()(tensor))[0, 0].numpy()
    return (probs > 0.5).astype(np.uint8), (time.perf_counter() - started) * 1000


def scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    inter = float(np.logical_and(pred, truth).sum())
    union = float(np.logical_or(pred, truth).sum())
    total = float(pred.sum() + truth.sum())
    return {
        "iou": round(inter / union, 4) if union else None,
        "dice": round(2 * inter / total, 4) if total else None,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "device": "cpu"}


@app.get("/samples")
def samples():
    return _manifest


@app.post("/segment/{sample_id}")
def segment_sample(sample_id: str):
    if sample_id not in _ids:
        raise HTTPException(status_code=404, detail="Unknown sample")
    cold = _model is None
    image = cv2.imread(str(SAMPLES / f"{sample_id}.jpg"), cv2.IMREAD_COLOR)
    truth = cv2.imread(str(SAMPLES / f"{sample_id}_mask.png"), cv2.IMREAD_GRAYSCALE)
    pred, ms = segment(image)
    truth_small = (
        cv2.resize(truth, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST) > 127
    ).astype(np.uint8)
    ok, png = cv2.imencode(".png", pred * 255)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode mask")
    return {
        "id": sample_id,
        "mask_png": base64.b64encode(png.tobytes()).decode(),
        "mask_size": IMG_SIZE,
        "inference_ms": round(ms),
        "polyp_area_pct": round(float(pred.mean()) * 100, 1),
        "model_was_cold": cold,
        "device": "cpu",
        **scores(pred, truth_small),
    }
