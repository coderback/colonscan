from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io
from PIL import Image

from xai import (
    CLASSES,
    load_model,
    get_patch_transforms,
    to_base64,
    infer_wsi,
    compute_gradcam_map,
    compute_saliency_map
)

app = FastAPI(title="ColonoScan Histopathology Service")


class PatchResponse(BaseModel):
    classification: str
    confidence: float
    saliency: str  # base64 PNG
    gradcam: str  # base64 PNG


@app.post("/infer/patch", response_model=PatchResponse)
async def infer_patch(file: UploadFile = File(...), patch_size: int = 224):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    model = load_model()
    tf = get_patch_transforms(patch_size)
    data = tf({"image": np.array(img)})
    inp = data["image"].unsqueeze(0).to(load_model().device)
    inp.requires_grad = True

    logits = model(inp)
    probs = F.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    label = CLASSES[idx]
    conf = float(probs[idx].item())

    sal_img = compute_saliency_map(img, idx, patch_size)
    cam_img = compute_gradcam_map(img, idx, patch_size)

    return PatchResponse(
        classification=label,
        confidence=conf,
        saliency=to_base64(sal_img),
        gradcam=to_base64(cam_img),
    )


@app.post("/infer/wsi")
async def infer_wsi_endpoint(path: str, patch_size: int = 224, overlap: float = 0.5):
    try:
        return JSONResponse({"result": infer_wsi(path, patch_size, overlap)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
