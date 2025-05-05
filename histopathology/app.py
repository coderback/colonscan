# histopathology/app.py
import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
import torchvision.transforms as T

# --- import your MONAI/ResNet50 utilities ---
from xai import (
    get_model,
    class_to_idx,
    compute_saliency_map,
    compute_gradcam_map,
    compute_shap_map,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model & weights
model = get_model(num_classes=len(class_to_idx))
model.load_state_dict(torch.load("/models/slide_model.pth", map_location=DEVICE))
model.to(DEVICE).eval()

# same transforms you used in training
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

app = FastAPI()

class InferRequest(BaseModel):
    path: str

class InferResponse(BaseModel):
    classification: str
    saliency: str   # base64‐PNG
    gradcam: str
    shap: str

def pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    try:
        img = Image.open(req.path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        # classification
        with torch.no_grad():
            logits = model(x)
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        label = {v:k for k,v in class_to_idx.items()}[pred]

        # four maps
        sal = compute_saliency_map(model, x, pred)    # returns PIL.Image
        cam = compute_gradcam_map(model, x, pred)
        shp = compute_shap_map(model, x, pred)

        return {
            "classification": label,
            "saliency": pil_to_b64(sal),
            "gradcam": pil_to_b64(cam),
            "shap": pil_to_b64(shp),
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))
