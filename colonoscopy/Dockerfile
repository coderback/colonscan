FROM python:3.13.2-slim-bullseye
WORKDIR /app

# 1) Install system build deps (needed for NumPy, OpenCV, etc.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      python3-dev \
      libgl1 \
 && rm -rf /var/lib/apt/lists/*

# 2) Install NumPy first
COPY requirements.txt ./
RUN pip install --no-cache-dir numpy

# 3) Install the rest of your requirements
RUN pip install --no-cache-dir -r requirements.txt

# 4) (Optional) purge build tools to slim the image
RUN apt-get purge -y --auto-remove \
      build-essential \
      gcc \
      python3-dev \
 && rm -rf /var/lib/apt/lists/*

# 5) Copy your app and set the startup command
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
