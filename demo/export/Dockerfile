FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

RUN pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install segmentation_models_pytorch==0.5.0 opencv-python-headless==4.11.0.86 onnx==1.18.0 onnxruntime==1.22.0

COPY export_model.py /app/export_model.py
WORKDIR /app
ENTRYPOINT ["python", "/app/export_model.py"]
