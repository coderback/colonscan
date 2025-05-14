# gradio_app.py

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import gradio as gr

# Configuration
MODEL_PATH = "../models/polyp-segmentation/efficient_unetpp.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
ROI = {"x1": 150, "y1": 90, "x2_offset": 150, "y2_offset": 90}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Initialize the segmentation model
def initialize_model():
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
    return model


model = initialize_model()


# Process a single frame
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


# Generator to stream frames from uploaded video

def stream_video(video_file):
    cap = cv2.VideoCapture(video_file.name)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_file.name}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield process_frame(frame)
    cap.release()


# Build a live, streaming interface without ffmpeg
grpc_interface = gr.Interface(
    fn=stream_video,
    inputs=gr.File(label="Upload MP4 video"),
    outputs=gr.Image(label="Segmented Frame"),
    live=True,
    title="Live-frame Polyp Segmentation",
    description="Upload a colonoscopy video and see frame-by-frame segmentation in real time."
)

if __name__ == "__main__":
    grpc_interface.launch()
