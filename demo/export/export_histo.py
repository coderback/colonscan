"""Export ColonoScan's histopathology patch classifier to ONNX for the demo page, then check it.

Rebuilds the test split of Jupyter-Notebook/Histopathology/monai-efficientnet-cnn.ipynb
from the public LC25000 file names, and stops if it doesn't match the notebook's saved
split sizes. It then scores the pinned weights on that split in PyTorch and in ONNX,
and writes eight test patches and their saved results for the page.
"""
import argparse
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import pyarrow.parquet as pq
import torch
from captum.attr import LayerGradCam
from monai.networks.nets import EfficientNetBN
from onnx import TensorProto, helper, numpy_helper
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

COMMIT = "1b96eec1444fcf26b49209b1c950c38eb7577ca6"
WEIGHTS_URL = (
    f"https://raw.githubusercontent.com/coderback/colonscan/{COMMIT}"
    "/models/tissue-classification/efficientnet-b3.pth"
)
WEIGHTS_SHA256 = "9837a1b416f793ba01f893bc5b5144262fa69f5b38eb22b4238fb4a3cff5d94c"
# The two parquet files that hold the colon images of LC25000.
DATA_URL = "https://huggingface.co/datasets/1aurent/LC25000/resolve/main/data/"
DATA_FILES = [
    "train-00002-of-00004-8b746b230324f3f3.parquet",
    "train-00003-of-00004-402d3c6cb6e6722a.parquet",
]
IMG_SIZE = 224
OPSET = 17
SAMPLES_PER_CLASS = 4

to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class WithSoftmax(torch.nn.Module):
    """The graph outputs probabilities, so the page only has to show them."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", path.name)
        urllib.request.urlretrieve(url, path)


def notebook_test_split() -> pd.DataFrame:
    """Repeats the notebook's two splits. It ran on Windows, where os.listdir sorts by name."""
    rows = []
    for fold, stem in (("colon_aca", "colonca"), ("colon_n", "colonn")):
        names = sorted(f"{stem}{i}.jpeg" for i in range(1, 5001))
        rows += [(name, fold) for name in names]
    df = pd.DataFrame(rows, columns=["name", "label"])
    train, dummy = train_test_split(df, train_size=0.8, random_state=123)
    valid, test = train_test_split(dummy, train_size=0.6, random_state=123)
    counts = test["label"].value_counts().to_dict()
    order = list(train["label"].unique())
    print("split sizes", len(train), len(valid), len(test), counts, "train order", order)
    if counts != {"colon_aca": 410, "colon_n": 390} or order != ["colon_n", "colon_aca"]:
        raise SystemExit("The rebuilt split doesn't match the notebook's saved output")
    return test


def read_images(cache: Path, wanted: set) -> dict:
    found = {}
    for name in DATA_FILES:
        path = cache / name
        fetch(DATA_URL + name, path)
        pf = pq.ParquetFile(path)
        for g in range(pf.metadata.num_row_groups):
            for row in pf.read_row_group(g, columns=["image", "organ"]).to_pylist():
                if row["organ"] == 1 and row["image"]["path"] in wanted:
                    found[row["image"]["path"]] = row["image"]["bytes"]
    missing = wanted - found.keys()
    if missing:
        raise SystemExit(f"{len(missing)} test images not found, for example {sorted(missing)[:3]}")
    return found


def load_model(weights: Path) -> torch.nn.Module:
    model = EfficientNetBN("efficientnet-b3", pretrained=False, spatial_dims=2, in_channels=3, num_classes=2)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    return model


def to_float16(src: Path, dst: Path) -> None:
    """Stores weights as float16 and casts them back at load time, as export_model.py does."""
    model = onnx.load(str(src))
    graph = model.graph
    casts, kept = [], []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims or [1]) < 2:
            kept.append(init)
            continue
        stored = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__f16")
        kept.append(stored)
        casts.append(helper.make_node("Cast", [stored.name], [init.name], to=TensorProto.FLOAT))
    del graph.initializer[:]
    graph.initializer.extend(kept)
    existing = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + existing)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def grad_cam(model: torch.nn.Module, pil: Image.Image, class_idx: int) -> Image.Image:
    """Same steps as compute_gradcam_map in histopathology/xai.py."""
    gradcam = LayerGradCam(model, model._conv_head)
    cam = gradcam.attribute(to_tensor(pil).unsqueeze(0), target=class_idx).squeeze(0).sum(dim=0)
    cam = torch.clamp(cam, min=0)
    cam = (cam - cam.min()) / (cam.max() + 1e-7)
    heat = cv2.applyColorMap((cam.detach().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)).resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    base = pil.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    return Image.blend(base.convert("RGB"), heat, alpha=0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="/demo")
    ap.add_argument("--cache", default="/work/cache")
    args = ap.parse_args()
    web, cache = Path(args.demo) / "web", Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    test = notebook_test_split()
    images = read_images(cache, set(test["name"]))

    weights = cache / "efficientnet-b3.pth"
    fetch(WEIGHTS_URL, weights)
    if sha256(weights) != WEIGHTS_SHA256:
        raise SystemExit("Weights checksum mismatch")
    model = load_model(weights)
    model.set_swish(memory_efficient=False)

    names = list(test["name"])
    truth = np.array([int(x == "colon_aca") for x in test["label"]])
    inputs = torch.stack([to_tensor(Image.open(io.BytesIO(images[n])).convert("RGB")) for n in names])

    with torch.no_grad():
        probs = torch.cat([torch.softmax(model(b), dim=1) for b in inputs.split(16)]).numpy()
    preds = probs.argmax(axis=1)
    correct = int((preds == truth).sum())
    print(f"PyTorch on the rebuilt test split: {correct} of {len(names)} correct")

    f32, f16 = cache / "histo.float32.onnx", cache / "histo.float16.onnx"
    torch.onnx.export(
        WithSoftmax(model).eval(), torch.zeros(1, 3, IMG_SIZE, IMG_SIZE), str(f32),
        input_names=["image"], output_names=["probability"], opset_version=OPSET,
        do_constant_folding=True, dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(f32)))
    to_float16(f32, f16)

    session = ort.InferenceSession(str(f16), providers=["CPUExecutionProvider"])
    onnx_probs = np.concatenate([session.run(None, {"image": x[None].numpy()})[0] for x in inputs])
    agree = int((onnx_probs.argmax(axis=1) == preds).sum())
    onnx_correct = int((onnx_probs.argmax(axis=1) == truth).sum())
    worst = float(np.abs(onnx_probs - probs).max())
    print(f"float16 ONNX: {onnx_correct} correct, agrees with PyTorch on {agree}, largest probability gap {worst:.6f}")
    if agree != len(names):
        raise SystemExit("float16 ONNX disagrees with PyTorch")
    confidence = probs.max(axis=1)

    digest = sha256(f16)
    served = web / "models" / f"efficientnet_b3.float16.{digest[:8]}.onnx"
    served.write_bytes(f16.read_bytes())
    (web / "models" / "histo.json").write_text(json.dumps({
        "file": served.name,
        "bytes": served.stat().st_size,
        "sha256": digest,
        "format": "float16",
        "release": f"demo-model-{digest[:8]}",
        "weights_commit": COMMIT,
    }, indent=1) + "\n")

    results = []
    samples = web / "samples"
    for cls, fold in enumerate(("colon_n", "colon_aca")):
        picks = [i for i, label in enumerate(test["label"]) if label == fold][:SAMPLES_PER_CLASS]
        for i in picks:
            n = len(results) + 1
            pil = Image.open(io.BytesIO(images[names[i]])).convert("RGB")
            (samples / f"histo-{n}.jpg").write_bytes(images[names[i]])
            pil.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR).save(samples / f"histo-{n}_224.png", optimize=True)
            grad_cam(model, pil, int(preds[i])).save(samples / f"histo-{n}_cam.jpg", quality=90)
            results.append({
                "id": f"histo-{n}",
                "source_file": names[i],
                "truth": cls,
                "probs": [round(float(p), 6) for p in probs[i]],
            })

    summary = {
        "test_total": len(names),
        "test_correct": correct,
        "onnx_correct": onnx_correct,
        "onnx_agree": agree,
        "max_probability_gap": round(worst, 6),
        "lowest_confidence": round(float(confidence.min()), 4),
        "under_99pct": int((confidence < 0.99).sum()),
    }
    (web / "histo-results.js").write_text(
        f"window.HISTO_RESULTS = {json.dumps(results)};\nwindow.HISTO_SUMMARY = {json.dumps(summary)};\n"
    )
    print(json.dumps(summary), served.name, served.stat().st_size)


if __name__ == "__main__":
    main()
