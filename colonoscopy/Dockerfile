FROM python:3.13.2-slim-bullseye

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py

# Copy models weights
COPY models/kvasir-yolov8-best.pt /models/kvasir-yolov8-best.pt

EXPOSE 8002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]