"""Export ColonoScan's polyp model to ONNX for the in-browser demo, then check it.

Reproduces the file served at demo/web/models/. Run it through the Dockerfile in
this folder (see README.md). It downloads the trained weights from a pinned
commit, exports them, writes size-reduced variants and compares every variant
with PyTorch on the eight public samples, using the same preprocessing and 0.5
threshold as demo/api/app.py.
"""
import argparse
import base64
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import segmentation_models_pytorch as smp
import torch
from onnx import TensorProto, helper, numpy_helper

COMMIT = "1b96eec1444fcf26b49209b1c950c38eb7577ca6"
WEIGHTS_URL = (
    f"https://raw.githubusercontent.com/coderback/colonscan/{COMMIT}"
    "/models/polyp-segmentation/efficient_unetpp.pth"
)
WEIGHTS_SHA256 = "dd656d5d86f0766bfa462a3f350e7a2d9e1db599bf0ae9a5883406848aa75007"
IMG_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
OPSET = 17


class WithSigmoid(torch.nn.Module):
    """The graph outputs probabilities, so the page only has to apply the threshold."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, path)
    digest = sha256(path)
    if digest != WEIGHTS_SHA256:
        raise SystemExit(f"Weights checksum mismatch: {digest}")


def load_model(weights: Path) -> torch.nn.Module:
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=1
    )
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    return model


def export_float32(model: torch.nn.Module, path: Path) -> None:
    # The exporter restores the wrapper's training flag on exit, and that flag
    # propagates to the model. A wrapper left in its default train mode would put
    # BatchNorm and drop-connect into training behaviour for everything after this.
    wrapper = WithSigmoid(model).eval()
    torch.onnx.export(
        wrapper,
        torch.zeros(1, 3, IMG_SIZE, IMG_SIZE),
        str(path),
        input_names=["image"],
        output_names=["probability"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))


def _prepend(graph, nodes) -> None:
    existing = list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes + existing)


def to_float16(src: Path, dst: Path) -> None:
    """Store weights as float16 and cast them back at load time.

    Only the stored weights change. All computation stays in float32, because the
    casts are constant-folded when the session starts, so inference speed matches
    the float32 model.
    """
    model = onnx.load(str(src))
    graph = model.graph
    casts, kept = [], []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims or [1]) < 2:
            kept.append(init)
            continue
        values = numpy_helper.to_array(init)
        name = init.name
        stored = numpy_helper.from_array(values.astype(np.float16), name + "__f16")
        kept.append(stored)
        casts.append(helper.make_node("Cast", [stored.name], [name], to=TensorProto.FLOAT))
    del graph.initializer[:]
    graph.initializer.extend(kept)
    _prepend(graph, casts)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def to_int8_weights(src: Path, dst: Path) -> None:
    """Store Conv weights as per-channel int8 and every other weight as float16."""
    model = onnx.load(str(src))
    graph = model.graph
    conv_weights = {n.input[1] for n in graph.node if n.op_type == "Conv"}
    added, kept = [], []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims or [1]) < 2:
            kept.append(init)
            continue
        values = numpy_helper.to_array(init)
        name = init.name
        if name in conv_weights:
            axes = tuple(range(1, values.ndim))
            scale = np.abs(values).max(axis=axes) / 127.0
            scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
            shape = (-1,) + (1,) * (values.ndim - 1)
            quant = np.clip(np.round(values / scale.reshape(shape)), -127, 127).astype(np.int8)
            kept += [
                numpy_helper.from_array(quant, name + "__q"),
                numpy_helper.from_array(scale, name + "__scale"),
                numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), name + "__zero"),
            ]
            added.append(
                helper.make_node(
                    "DequantizeLinear",
                    [name + "__q", name + "__scale", name + "__zero"],
                    [name],
                    axis=0,
                )
            )
        else:
            kept.append(numpy_helper.from_array(values.astype(np.float16), name + "__f16"))
            added.append(helper.make_node("Cast", [name + "__f16"], [name], to=TensorProto.FLOAT))
    del graph.initializer[:]
    graph.initializer.extend(kept)
    _prepend(graph, added)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def preprocess(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same steps as demo/api/app.py. Returns the 256x256 RGB image and the tensor."""
    resized = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb8 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb8.astype(np.float32) / 255.0
    return rgb8, ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]


def scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    inter = float(np.logical_and(pred, truth).sum())
    union = float(np.logical_or(pred, truth).sum())
    total = float(pred.sum() + truth.sum())
    return {
        "iou": round(inter / union, 4) if union else None,
        "dice": round(2 * inter / total, 4) if total else None,
    }


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def compare(model, variants: dict, samples: Path, web: Path, report: Path) -> None:
    sessions = {
        name: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        for name, p in variants.items()
    }
    manifest = json.loads((samples / "manifest.json").read_text())
    saved_js = (web / "results.js").read_text()
    saved = {r["id"]: r for r in json.loads(saved_js[saved_js.index("["):saved_js.rindex("]") + 1])}
    table, baseline, stress = [], {}, {name: [] for name in variants}
    for entry in manifest:
        sid = entry["id"]
        bgr = cv2.imread(str(samples / f"{sid}.jpg"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(samples / f"{sid}_mask.png"), cv2.IMREAD_GRAYSCALE)
        truth = (
            cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST) > 127
        ).astype(np.uint8)
        rgb8, tensor = preprocess(bgr)
        with torch.inference_mode():
            probs = torch.sigmoid(model(torch.from_numpy(tensor)))[0, 0].numpy()
        pred = (probs > 0.5).astype(np.uint8)
        row = {"id": sid, "pytorch": scores(pred, truth), "area_pct": round(float(pred.mean()) * 100, 1)}
        # The baseline has to reproduce the saved results before it can judge anything.
        for key in ("iou", "dice"):
            if row["pytorch"][key] != saved[sid][key]:
                raise SystemExit(f"{sid} {key}: PyTorch {row['pytorch'][key]}, saved {saved[sid][key]}")
        for name, sess in sessions.items():
            p = sess.run(None, {"image": tensor})[0][0, 0]
            m = (p > 0.5).astype(np.uint8)
            row[name] = {
                "mask_iou_vs_pytorch": round(mask_iou(m, pred), 5),
                "pixels_changed": int((m != pred).sum()),
                "max_abs_prob_diff": float(np.abs(p - probs).max()),
                **scores(m, truth),
            }
        table.append(row)
        baseline[sid] = {
            "mask_b64": base64.b64encode(pred.tobytes()).decode(),
            "truth_b64": base64.b64encode(truth.tobytes()).decode(),
            "rgb_b64": base64.b64encode(rgb8.tobytes()).decode(),
            "iou": row["pytorch"]["iou"],
            "dice": row["pytorch"]["dice"],
            "area_pct": row["area_pct"],
        }
        ok, png = cv2.imencode(".png", truth * 255)
        assert ok
        (web / "samples" / f"{sid}_truth256.png").write_bytes(png.tobytes())
        # Extra inputs the samples don't cover, to test size-reduced weights harder.
        for flip in ("h", "v", "r"):
            alt = {"h": bgr[:, ::-1], "v": bgr[::-1], "r": np.rot90(bgr)}[flip].copy()
            _, t = preprocess(alt)
            with torch.inference_mode():
                q = torch.sigmoid(model(torch.from_numpy(t)))[0, 0].numpy()
            for name, sess in sessions.items():
                stress[name].append(mask_iou(sess.run(None, {"image": t})[0][0, 0] > 0.5, q > 0.5))
    out = {
        "samples": table,
        "transformed_inputs_min_mask_iou": {k: round(min(v), 5) for k, v in stress.items()},
        "sizes_bytes": {k: p.stat().st_size for k, p in variants.items()},
    }
    (report / "parity_python.json").write_text(json.dumps(out, indent=1))
    (report / "baseline.json").write_text(json.dumps(baseline))
    print(json.dumps(out["sizes_bytes"]), json.dumps(out["transformed_inputs_min_mask_iou"]))
    for row in table:
        cells = " ".join(
            f"{n}={row[n]['mask_iou_vs_pytorch']:.4f}/{row[n]['iou']:.4f}" for n in variants
        )
        print(row["id"], f"pytorch={row['pytorch']['iou']:.4f}/{row['pytorch']['dice']:.4f}", cells)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("/work/weights/efficient_unetpp.pth"))
    ap.add_argument("--samples", type=Path, default=Path("/demo/api/samples"))
    ap.add_argument("--web", type=Path, default=Path("/demo/web"))
    ap.add_argument("--out", type=Path, default=Path("/work/out"))
    ap.add_argument("--ship", default="float16", choices=["float32", "float16", "int8"])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fetch_weights(args.weights)
    torch.set_num_threads(4)
    model = load_model(args.weights)

    variants = {
        "float32": args.out / "model.float32.onnx",
        "float16": args.out / "model.float16.onnx",
        "int8": args.out / "model.int8.onnx",
    }
    export_float32(model, variants["float32"])
    to_float16(variants["float32"], variants["float16"])
    to_int8_weights(variants["float32"], variants["int8"])
    compare(model, variants, args.samples, args.web, args.out)

    # The checksum in the name lets the file be cached forever.
    digest = sha256(variants[args.ship])
    target = args.web / "models" / f"efficient_unetpp.{args.ship}.{digest[:8]}.onnx"
    target.parent.mkdir(exist_ok=True)
    for old in target.parent.glob("*.onnx"):
        old.unlink()
    shutil.copyfile(variants[args.ship], target)
    info = {
        "file": target.name,
        "bytes": target.stat().st_size,
        "sha256": digest,
        "format": args.ship,
        "weights_commit": COMMIT,
    }
    (target.parent / "model.json").write_text(json.dumps(info, indent=1) + "\n")
    print("shipped", info)


if __name__ == "__main__":
    main()
