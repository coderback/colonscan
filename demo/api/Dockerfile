FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

WORKDIR /app

COPY requirements.txt .
RUN pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# The trained weights already live in the repo. Fetching them from the pinned
# commit keeps this deployment's upload small and avoids a second copy in git.
ADD https://raw.githubusercontent.com/coderback/colonscan/1b96eec1444fcf26b49209b1c950c38eb7577ca6/models/polyp-segmentation/efficient_unetpp.pth /app/efficient_unetpp.pth

COPY app.py /app/app.py
COPY samples /app/samples

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
