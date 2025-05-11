FROM python:3.13.2-slim-bullseye
WORKDIR /app

# 1) Install build-time and runtime dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      python3-dev \
      libgl1 \
 && rm -rf /var/lib/apt/lists/*

# 2) Install Python packages (NumPy first for caching and fast failure)
COPY requirements.txt ./
RUN pip install --no-cache-dir numpy \
 && pip install --no-cache-dir -r requirements.txt

# 3) Remove build dependencies to slim down the image
RUN apt-get purge -y --auto-remove \
      build-essential \
      gcc \
      python3-dev \
 && rm -rf /var/lib/apt/lists/*

# 4) Copy application code
COPY . .

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
