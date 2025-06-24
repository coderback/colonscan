import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional

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
    infer_wsi_with_heatmap,
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
    summary: str
    overview_map: Optional[str] = None  # base64 PNG heatmap


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
        include_heatmap: bool = Query(True, description="Generate overview heatmap"),
):
    """Whole‐slide inference (returns mean score summary and optional heatmap)."""
    print(f"Received slide analysis request: {file.filename}, size: {file.size}")
    
    try:
        contents = await file.read()
        print(f"Read {len(contents)} bytes from uploaded file")
        
        # save upload to temp file
        tmp = "/tmp/slide.svs"
        with open(tmp, "wb") as f:
            f.write(contents)
        print(f"Saved slide to {tmp}")
        
    except Exception as e:
        print(f"Error saving slide: {e}")
        raise HTTPException(400, f"Unable to save slide: {str(e)}")

    try:
        print("Starting slide analysis...")
        if include_heatmap:
            summary, heatmap_img = infer_wsi_with_heatmap(tmp, patch_size=patch_size, overlap=overlap)
            print(f"Analysis completed with heatmap. Summary: {summary}")
            return SlideResult(
                summary=summary,
                overview_map=to_base64(heatmap_img) if heatmap_img else None
            )
        else:
            summary = infer_wsi(tmp, patch_size=patch_size, overlap=overlap)
            print(f"Analysis completed. Summary: {summary}")
            return SlideResult(summary=summary)
    except Exception as e:
        print(f"Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}
