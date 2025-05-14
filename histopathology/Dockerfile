FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# Install any OS-level deps you still need
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    libgl1 \
    libopenslide-dev \
    openslide-tools \
 && rm -rf /var/lib/apt/lists/*

# Install the rest of your Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
