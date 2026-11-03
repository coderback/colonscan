# Browser model export

Builds the ONNX file that the demo page runs in the visitor's browser. The file is 40 MB, so it stays out of git (`demo/web/.gitignore`) and is uploaded with the site instead. Run this before deploying from a fresh clone, or the live control will report that it can't find the model.

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
