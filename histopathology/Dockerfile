# histopathology/Dockerfile

# 1) Build stage: install compilers, build everything
FROM python:3.13-slim-bullseye AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc python3-dev libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# 2) Final runtime stage: only pull in the wheels & your code
FROM python:3.13-slim-bullseye

# Only need the minimal C libs for pillow/OpenCV
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Copy in all the site-packages from the builder
COPY --from=builder /install /usr/local

WORKDIR /app
COPY app.py xai.py /app/
COPY models/slide_model.pth /models/slide_model.pth

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
