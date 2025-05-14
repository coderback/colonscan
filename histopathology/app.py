import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict

from PIL import Image
import torch

from xai import (
    load_model,
    DEVICE,
    CLASSES,
    get_patch_transforms,
    compute_gradcam_map,
    compute_saliency_map,
    infer_wsi,
    to_base64,
)

app = FastAPI(title="Colon Histopathology Inference API")

MODEL = load_model()


class PatchResult(BaseModel):
    predicted_class: int
    class_name: str
    probabilities: List[float]
    gradcam: str  # base64 PNG
    saliency: str  # base64 PNG


class SlideResult(BaseModel):
    slide_summary: str


@app.post("/infer/patch", response_model=List[PatchResult])
async def infer_patch(
    files: List[UploadFile] = File(..., description="One or more image files"),
    patch_size: int = Query(224, ge=32, le=1024),
):
    """
    Patch‐level classification + GradCAM + saliency maps on one or more images.
    """
    results = []

    for file in files:
        contents = await file.read()
        try:
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(400, f"Unable to read image {file.filename}")

        # 1) Classify
        tf = get_patch_transforms(patch_size)
        inp = tf(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = MODEL(inp)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
            idx = int(max(range(len(probs)), key=lambda i: probs[i]))

        # 2) XAI overlays
        gradcam_img = compute_gradcam_map(pil_img, idx, patch_size)
        saliency_img = compute_saliency_map(pil_img, idx, patch_size)

        results.append(
            PatchResult(
                predicted_class=idx,
                class_name=CLASSES[idx],
                probabilities=probs,
                gradcam=to_base64(gradcam_img),
                saliency=to_base64(saliency_img),
            )
        )

    return results


@app.post("/infer/slide", response_model=SlideResult)
async def infer_slide(
        file: UploadFile = File(...),
        patch_size: int = Query(224, ge=32, le=1024),
        overlap: float = Query(0.5, ge=0.0, le=1.0),
):
    """Whole‐slide inference (returns mean score summary)."""
    contents = await file.read()
    try:
        # save upload to temp file
        tmp = "/tmp/slide.svs"
        with open(tmp, "wb") as f:
            f.write(contents)
    except Exception:
        raise HTTPException(400, "Unable to save slide")

    summary = infer_wsi(tmp, patch_size=patch_size, overlap=overlap)
    return SlideResult(slide_summary=summary)


@app.get("/health")
async def health():
    return {"status": "ok"}
