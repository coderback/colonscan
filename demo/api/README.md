---
title: ColonoScan Polyp Segmentation Demo
sdk: docker
app_port: 8080
pinned: false
---

# ColonoScan demo inference service

Research demo, not for clinical use.

Runs the repo's trained EfficientNet-b4 UNet++ on CPU against a fixed list of public Kvasir-SEG images. It has no upload endpoint.

- `GET /health`
- `GET /samples`
- `POST /segment/{sample_id}`

Local run: `docker build -t colonoscan-demo-api . && docker run -p 8080:8080 colonoscan-demo-api`

The saved results on the demo page came from this image running with 2 CPU cores and a 2 GB memory limit.
