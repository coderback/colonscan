import io
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, Response
from starlette.responses import Response
import tempfile
import time
import base64
from fastapi.middleware.cors import CORSMiddleware

# Model config
MODEL_PATH = "models/polyp_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
ROI = {"x1": 150, "y1": 90, "x2_offset": 150, "y2_offset": 90}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
@app.on_event("startup")
def load_model():
    global model
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

# Helper: process a single frame
def process_frame(frame: np.ndarray) -> np.ndarray:
    h, w, _ = frame.shape
    x1, y1 = ROI["x1"], ROI["y1"]
    x2, y2 = w - ROI["x2_offset"], h - ROI["y2_offset"]
    crop = frame[y1:y2, x1:x2]
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
    norm = (rgb - MEAN) / STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        mask = (torch.sigmoid(out) > 0.5).cpu().numpy()[0, 0].astype(np.uint8)
    mask_full = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    overlay = np.zeros_like(frame)
    overlay[y1:y2, x1:x2, 2] = mask_full * 255
    blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    return blended

@app.post("/detect")
def detect_video(file: UploadFile = File(...)):
    # Save uploaded file to a temp location
    try:
        contents = file.file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        # Write to temp file for OpenCV
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_in, tempfile.NamedTemporaryFile(suffix=".mp4") as temp_out:
            temp_in.write(contents)
            temp_in.flush()
            cap = cv2.VideoCapture(temp_in.name)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open uploaded video.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            x1, y1 = ROI["x1"], ROI["y1"]
            x2, y2 = width - ROI["x2_offset"], height - ROI["y2_offset"]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                blended = process_frame(frame)
                writer.write(blended)
            cap.release()
            writer.release()
            temp_out.seek(0)
            video_bytes = temp_out.read()
            return Response(content=video_bytes, media_type="video/mp4", headers={"Content-Disposition": f"attachment; filename=processed_{file.filename}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-frame")
def detect_frame(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        print("Frame shape:", frame.shape if frame is not None else None)
        if frame is None:
            print("Invalid image file received.")
            raise HTTPException(status_code=400, detail="Invalid image file.")
        blended = process_frame(frame)
        success, img_encoded = cv2.imencode('.jpg', blended)
        print("Encode success:", success, "Encoded size:", len(img_encoded) if success else None)
        if not success:
            print("Failed to encode image.")
            raise HTTPException(status_code=500, detail="Failed to encode image.")
        return Response(content=img_encoded.tobytes(), media_type="image/jpeg")
    except Exception as e:
        print("Segmentation error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream-segmentation")
def stream_video_segmentation(file: UploadFile = File(...)):
    """
    Stream video segmentation in real-time using Server-Sent Events.
    Returns frames one by one as they are processed, similar to Gradio's approach.
    """
    try:
        print(f"Starting streaming for file: {file.filename}")
        contents = file.file.read()
        print(f"File size: {len(contents)} bytes")
        
        # Create temporary file for video processing
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(contents)
            temp_file.flush()
            
            cap = cv2.VideoCapture(temp_file.name)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open uploaded video.")
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            frame_delay = 1.0 / fps  # Time between frames
            print(f"Video FPS: {fps}, Frame delay: {frame_delay}s")
            
            def generate_frames():
                frame_count = 0
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            print(f"End of video reached after {frame_count} frames")
                            break
                        
                        # Process the frame
                        processed_frame = process_frame(frame)
                        
                        # Encode as JPEG
                        success, img_encoded = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if not success:
                            print(f"Failed to encode frame {frame_count}")
                            continue
                        
                        # Convert to base64 for SSE
                        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
                        frame_count += 1
                        
                        if frame_count % 10 == 0:  # Log every 10th frame
                            print(f"Processed frame {frame_count}, image size: {len(img_base64)} chars")
                        
                        # Send SSE event
                        yield f"data: {{\"frame\": {frame_count}, \"image\": \"data:image/jpeg;base64,{img_base64}\"}}\n\n"
                        
                        # Small delay to control frame rate
                        time.sleep(frame_delay)
                finally:
                    cap.release()
                    print("Video capture released")
            
            return StreamingResponse(
                generate_frames(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Methods": "*"
                }
            )
            
    except Exception as e:
        print(f"Streaming error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))