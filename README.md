# ColonoScan

An end-to-end, containerized platform for automated analysis of colonoscopy data: histopathology slides, colonoscopy videos, and genomic samples. Built with a Django REST backend, Next.js frontend, and specialized microservices for deep-learning inference (FastAPI + PyTorch/TensorFlow + YOLOv5) and genomic pipeline execution (Nextflow or Flask).

---

## Table of Contents

- [Features](#features)  
- [Architecture](#architecture)  
- [Prerequisites](#prerequisites)  
- [Getting Started](#getting-started)  
- [Usage](#usage)  
  - [Health Check](#health-check)  
  - [Slides API](#slides-api)  
  - [Videos API](#videos-api)  
  - [Genomic API](#genomic-api)  
  - [Patches API (Grad-CAM & Saliency)](#patches-api-grad-cam--saliency)  
  - [Jobs API](#jobs-api)  
- [Development](#development)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Features

- **Histopathology Slide Analysis**  
  - Upload H&E-stained WSIs  
  - Receive per-slide segmentation masks & Grad-CAM heatmaps  
  - Powered by a MONAI-based ResNet50 CNN service  

- **Colonoscopy Video Analysis**  
  - Upload colonoscopy video  
  - Receive detected polyp bounding boxes & counts  
  - Powered by YOLOv5 inference service  

- **Genomic Sample Profiling**  
  - Upload FASTQ/DNA samples  
  - Receive variant calls (VCF) & QC metrics  

- **Interpretability Maps**  
  - Grad-CAM & saliency overlays for uploaded image “patches”

- **Asynchronous Processing**  
  - Celery + Redis for job queuing  
  - PostgreSQL persistence  

---

## Architecture

\`\`\`
┌────────────┐     ┌─────────────┐     ┌──────────────┐
│  Frontend  │◀───▶│  Backend    │◀───▶│ PostgreSQL   │
│ (Next.js)  │     │ (Django +   │     │ + Django ORM │
└────────────┘     │  DRF + Celery)   └──────────────┘
                       ▲     ▲     ▲
                       │     │     │
        ┌──────────────┘     │     └─────────────┐
        │                    │                   │
┌─────────────  ┐       ┌─────────────────┐    ┌──────────────┐
│ Histopath.    │       │ Colonoscopy     │    │ Genomic      │
│ Service       │       │ Service         │    │ Profiling    │
│ (FastAPI +    │       │ (FastAPI +      │    │ Service      │
│ MONAI/
 EfficientNet)    │       │ EfficientUnet++
                          )       │       │      (Nextflow/   │
└─────────────  ┘       └─────────────────┘    │  Flask)      │
                                             └──────────────┘
\`\`\`

- **Ports**  
  - Backend: \`8000\`  
  - Frontend: \`3000\`  
  - Histopathology: \`8001\`  
  - Colonoscopy: \`8002\`  
  - Genomic Profiling: \`8003\`  
  - Redis: \`6379\`  
  - Postgres: \`5432\`

---

## Prerequisites

- [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)  
- (Optional) NVIDIA GPU + [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) for accelerated inference  

---

## Getting Started

1. **Clone the repo**  
   \`\`\`bash
   git clone https://github.com/your-org/colonscan.git
   cd colonscan
   \`\`\`

2. **Create a \`.env\` file** (optional – defaults are in \`docker-compose.yml\`)
   \`\`\`ini
   POSTGRES_DB=colonscan_db
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=admin
   DATABASE_HOST=db
   DATABASE_PORT=5432
   REDIS_URL=redis://redis:6379/0
   \`\`\`

3. **Build & launch all services**  
   \`\`\`bash
   docker-compose up --build
   \`\`\`

4. **Run migrations & create superuser**  
   \`\`\`bash
   docker-compose exec backend python manage.py migrate
   docker-compose exec backend python manage.py createsuperuser
   \`\`\`

5. **Browse**  
   - Frontend:  http://localhost:3000  
   - API root:  http://localhost:8000/api/  

---

## Usage

### Health Check

\`\`\`bash
GET /api/health/
→ { "status": "ColonoScan backend is up and running" }
\`\`\`

---

### Slides API

\`\`\`bash
POST   /api/slides/      # upload a \`.ndpi\` or \`.tiff\`
GET    /api/slides/      # list your uploads & statuses
GET    /api/slides/{id}/ # retrieve detail with status & result_url
\`\`\`

- **Response fields**  
  - \`status\`: PENDING → RUNNING → SUCCESS / FAILED  
  - \`result_url\`: downloadable segmentation mask  

---

### Videos API

\`\`\`bash
POST   /api/videos/         # upload \`.mp4\` or \`.avi\`
GET    /api/videos/         # list jobs
GET    /api/videos/{id}/    # get JSON result_data
\`\`\`

- **Fields**  
  - \`frame_rate\`, \`resolution\`, \`uploaded\`, \`status\`  
  - \`result\`: contains \`polyp_count\`, bounding boxes, timestamps  

---

### Genomic API

\`\`\`bash
POST   /api/genomic/        # upload FASTQ(s)
GET    /api/genomic/        # list jobs
GET    /api/genomic/{id}/   # get VCF download & metrics
\`\`\`

- **Fields**  
  - \`vcf\` (URL), \`metrics\` JSON  

---

### Patches API (Grad-CAM & Saliency)

\`\`\`bash
POST   /api/patches/        # upload a crop/patch (JPEG/PNG)
GET    /api/patches/        # list patches
GET    /api/patches/{id}/   # get \`predicted_class\`, \`probabilities\`, \`gradcam_url\`, \`saliency_url\`
\`\`\`

---

### Jobs API

\`\`\`bash
GET /api/jobs/             # view all AnalysisJob statuses
\`\`\`

---

## Development

- **Backend**  
  \`\`\`bash
  docker-compose exec backend bash
  cd core
  pytest
  flake8
  \`\`\`

- **Frontend**  
  \`\`\`bash
  cd frontend
  npm install
  npm run dev
  \`\`\`

- **Microservices**  
  - Histopathology: \`cd histopathology && uvicorn main:app --reload --port 8001\`  
  - Colonoscopy:   \`cd colonoscopy   && uvicorn main:app --reload --port 8002\`  
  - Genomics:      \`cd genomic_profiling && flask run --port 8003\` (or \`nextflow run ...\`)

---

## Contributing

1. Fork the repo  
2. Create a feature branch (\`git checkout -b feat/my-feature\`)  
3. Commit & push (\`git push origin feat/my-feature\`)  
4. Open a Pull Request  

Please follow our code style (PEP8 / ESLint) and include tests for new features.

---

## License

This project is licensed under the **MIT License**.  
See [LICENSE](LICENSE) for details.
