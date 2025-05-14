# ── builder stage: build all wheels ─────────────────────────────────────────
FROM python:3.13.2-slim-bullseye AS builder
WORKDIR /build

# install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc python3-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# build wheels for all dependencies
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

# ── final stage: runtime only ────────────────────────────────────────────────
FROM python:3.13.2-slim-bullseye
WORKDIR /app

# runtime deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 \
 && rm -rf /var/lib/apt/lists/*

# install the pre-built wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-index --no-cache-dir /wheels/*.whl \
 && rm -rf /wheels

COPY . .
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
