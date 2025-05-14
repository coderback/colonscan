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


# Load the FPN model
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


def segment_video(video_path: str) -> str:
    """
    Process an input MP4 video via OpenCV, detect polyps frame-by-frame,
    overlay segmentation masks, and save the output to 'output.mp4'.
    Returns the path to the processed video file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    # Define ROI boundaries
    x1, y1 = ROI["x1"], ROI["y1"]
    x2 = width - ROI["x2_offset"]
    y2 = height - ROI["y2_offset"]

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = "output.mp4"
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop ROI from frame
        crop = frame[y1:y2, x1:x2]
        # Preprocess for model input
        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        rgb_norm = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
        normalized = (rgb_norm - MEAN) / STD
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

        # Inference
        with torch.no_grad():
            output = model(tensor)
            mask = (torch.sigmoid(output) > 0.5).cpu().numpy()[0, 0].astype(np.uint8)

        # Resize mask to original crop size
        mask_full = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        overlay = np.zeros_like(frame)
        overlay[y1:y2, x1:x2, 2] = mask_full * 255

        # Blend overlay and write frame
        blended_frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        writer.write(blended_frame)

    cap.release()
    writer.release()
    return out_path


# Build Gradio interface using File components to avoid ffmpeg dependency
iface = gr.Interface(
    fn=segment_video,
    inputs=gr.File(label="Upload colonoscopy video (MP4)"),
    outputs=gr.File(label="Processed video (MP4)"),
    title="Polyp Segmentation Demo",
    description="Upload an MP4 colonoscopy video; receive an annotated MP4 with red polyp masks."
)

if __name__ == "__main__":
    iface.launch()
