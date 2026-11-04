# Browser model export

Builds the ONNX files that the demo page runs in the visitor's browser. They are too large for git (`demo/web/.gitignore`), so each one is published as a GitHub release, and `demo/fetch-model.sh` downloads it at build time and checks its SHA-256. To change a model, publish a new release and update `release` in its JSON file in `demo/web/models`. The histopathology classifier has its own export, described at the end.

From the repo root, with Docker running:

```sh
docker build -t colonoscan-export demo/export
docker run --rm \
  -v "$PWD/demo:/demo" \
  -v "$PWD/.export-cache/weights:/work/weights" \
  -v "$PWD/.export-cache/out:/work/out" \
  colonoscan-export --ship float16
```

It downloads the weights from a pinned commit and checks their SHA-256, then exports them and writes:

- `demo/web/models/efficient_unetpp.float16.<hash>.onnx`, the served model
- `demo/web/models/model.json`, its size and SHA-256, which the page checks after downloading
- `demo/web/samples/*_truth256.png`, the expert masks at the model's resolution
- `parity_python.json` in the output folder, comparing PyTorch with each ONNX variant

The script stops if PyTorch doesn't reproduce the saved IoU and Dice in `demo/web/results.js`.

Delete `.export-cache` afterwards, and don't commit it.

## Why float16

Compared with PyTorch in onnxruntime 1.22 on the eight samples, plus each sample flipped both ways and rotated (24 more inputs):

| Variant | Size | Lowest mask IoU, samples | Lowest mask IoU, transformed |
| --- | --- | --- | --- |
| float32 | 80.1 MB | 1.0000 | 1.0000 |
| float16 weights | 40.2 MB | 0.9993 | 0.9982 |
| int8 Conv weights | 20.8 MB | 0.9925 | 0.8819 |

The int8 version clears 0.99 on the samples but not on the transformed inputs, so it wasn't used. Only the stored weights are float16. They are cast back when the model loads, so all computation is float32.

## Preprocessing in the browser

A canvas resize uses a different filter from OpenCV, so `demo/web/live.js` does the resize itself, using the same fixed-point arithmetic as `cv2.resize` with `INTER_LINEAR`. In Chromium, Firefox and WebKit builds, the resized pixels matched `cv2` exactly on all eight samples.

## Histopathology classifier

From the repo root, with Docker running:

```sh
docker build -t colonoscan-histo -f demo/export/Dockerfile.histo demo/export
docker run --rm -v "$PWD/demo:/demo" -v "$PWD/.export-cache/histo:/work/cache" colonoscan-histo
```

It downloads about 1.1 GB of LC25000 from Hugging Face and the weights from a pinned commit. It repeats the two splits in `monai-efficientnet-cnn.ipynb` from the file names, and stops if the test split does not have the 390 benign and 410 adenocarcinoma patches the notebook reports. It then scores those 800 patches in PyTorch and in float16 ONNX, and stops if the two disagree on any label. It writes:

- `demo/web/models/efficientnet_b3.float16.<hash>.onnx`, the served model
- `demo/web/models/histo.json`, its size, SHA-256 and release
- `demo/web/samples/histo-*`, eight test patches, each with the 224 pixel input and a saved Grad-CAM overlay
- `demo/web/histo-results.js`, their saved results and the test scores

The page reads the 224 pixel PNG, so its input matches the PIL resize in PyTorch with no browser resize involved.
