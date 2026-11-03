# ColonoScan demo inference service

Research demo, not for clinical use.

Runs the repo's trained EfficientNet-b4 UNet++ on CPU against a fixed list of public Kvasir-SEG images. It has no upload endpoint.

**This service is not deployed, and the public demo page does not call it.** The page replays saved results and can run the model in the visitor's browser (see `demo/export`). Hugging Face now needs a paid plan for a Docker Space, so this service was not hosted there.

- `GET /health`
- `GET /samples`
- `POST /segment/{sample_id}`

Local run: `docker build -t colonoscan-demo-api . && docker run -p 8080:8080 colonoscan-demo-api`

The saved results on the demo page came from this image running with 2 CPU cores and a 2 GB memory limit. Its preprocessing, 256 by 256 size and 0.5 threshold are the reference for the browser version.
