import io
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def create_dummy_video():
    # Create a short dummy video in memory
    height, width = 256, 256
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('dummy.mp4', fourcc, 5, (width, height))
    for _ in range(5):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    with open('dummy.mp4', 'rb') as f:
        return io.BytesIO(f.read())

def create_dummy_image():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', img)
    return io.BytesIO(buf.tobytes())

def test_detect_video():
    video = create_dummy_video()
    response = client.post("/detect", files={"file": ("test.mp4", video, "video/mp4")})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")

def test_detect_frame():
    img = create_dummy_image()
    response = client.post("/detect-frame", files={"file": ("test.jpg", img, "image/jpeg")})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")

def test_detect_video_invalid():
    response = client.post("/detect", files={"file": ("bad.txt", io.BytesIO(b"not a video"), "text/plain")})
    assert response.status_code == 500 or response.status_code == 400

def test_detect_frame_invalid():
    response = client.post("/detect-frame", files={"file": ("bad.txt", io.BytesIO(b"not an image"), "text/plain")})
    assert response.status_code == 500 or response.status_code == 400 