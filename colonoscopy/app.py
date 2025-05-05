from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import torch

# Load your custom YOLO models
model = torch.hub.load('ultralytics/yolov8', 'custom', path='/models/kvasir-yolov8-best.pt', source='local')
model.eval()

class FrameRequest(BaseModel):
    frame: str  # base64 JPEG

class DetectResponse(BaseModel):
    detections: list

app = FastAPI()

@app.post("/detect-frame", response_model=DetectResponse)
def detect_frame(req: FrameRequest):
    try:
        b = base64.b64decode(req.frame)
        arr = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        results = model(img)
        dets = [
            {
                "box": [float(x) for x in row[:4]],
                "score": float(row[4]),
                "class": int(row[5])
            }
            for row in results.xyxy[0].tolist()
        ]
        return {"detections": dets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))