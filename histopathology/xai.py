import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from monai.networks.nets import EfficientNetBN
from torchvision import transforms
from captum.attr import LayerGradCam
from monai.data import WSIReader, SlidingPatchWSIDataset, list_data_collate
from monai.transforms import (
    LoadImageD,
    EnsureChannelFirstD,
    ScaleIntensityRangeD,
    ResizeD,
    ToTensord,
    Compose,
)

# ---- Configuration ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["benign", "malignant"]
MODEL_PATH = "models/slide_model.pth"


# ---- Model loading ----
def load_model():
    if not hasattr(load_model, "model"):
        model = EfficientNetBN(
            spatial_dims=2,
            in_channels=3,
            model_name="efficientnet-b3",
            pretrained=True,
            num_classes=len(CLASSES),
        ).to(DEVICE)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        load_model.model = model
    return load_model.model


# ---- Preprocessing ----
def get_patch_transforms(patch_size: int):
    return transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# Slide-level transforms (window-by-window reuse the same patch transforms)
def get_slide_transforms(patch_size: int):
    return Compose([
        # your patch already comes as a H×W×C NumPy/MetaTensor
        EnsureChannelFirstD(keys=["image"]),                      # → C×H×W
        ScaleIntensityRangeD(
            keys=["image"],
            a_min=0.0, a_max=255.0,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),                                                         # → [0–1]
        ResizeD(keys=["image"], spatial_size=[patch_size, patch_size]),
        ToTensord(keys=["image"]),                                # → torch.Tensor C×H×W
    ])


# ---- Utility ----
def to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# Grad-CAM map computation
def compute_gradcam_map(pil_img: Image.Image, class_idx: int, patch_size: int) -> Image.Image:
    model = load_model().to(DEVICE)
    model.eval()

    # transform
    tf = get_patch_transforms(patch_size)
    inp = tf(pil_img).unsqueeze(0).to(DEVICE)

    # get attributions
    gradcam = LayerGradCam(model, model._conv_head)
    attributions = gradcam.attribute(inp, target=class_idx)

    # aggregate and normalize
    cam = attributions.squeeze(0).sum(dim=0)
    cam = torch.clamp(cam, min=0)
    cam = (cam - cam.min()) / (cam.max() + 1e-7)
    cam_np = (cam.cpu().detach().numpy() * 255).astype(np.uint8)

    # build heatmap image
    heatmap = cv2.applyColorMap(cam_np, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap_img = Image.fromarray(heatmap).resize((patch_size, patch_size), resample=Image.BILINEAR)

    # resize the original to patch_size×patch_size
    base = pil_img.resize((patch_size, patch_size), resample=Image.BILINEAR)

    # blend
    overlay = Image.blend(base.convert("RGB"), heatmap_img, alpha=0.5)
    return overlay


# Saliency map computation
def compute_saliency_map(pil_img: Image.Image, class_idx: int, patch_size: int) -> Image.Image:
    model = load_model().to(DEVICE)
    model.eval()

    tf = get_patch_transforms(patch_size)
    inp = tf(pil_img).unsqueeze(0).to(DEVICE)
    inp.requires_grad_(True)

    out = model(inp)
    score = out[0, class_idx]
    score.backward()

    saliency, _ = torch.max(inp.grad.abs(), dim=1)
    saliency = saliency.squeeze(0).cpu().detach().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
    saliency = np.uint8(255 * saliency)
    saliency_img = Image.fromarray(saliency).resize((patch_size, patch_size)).convert("RGB")
    return saliency_img


# ---- WSI inference ----
def infer_wsi(slide_path: str, patch_size: int = 224, overlap: float = 0.5) -> str:
    reader = WSIReader("openslide")
    ds = SlidingPatchWSIDataset(
        data=[{"image": slide_path}],
        patch_size=patch_size,
        overlap=overlap,
        transform=get_slide_transforms(patch_size),
        reader=reader,
        center_location=False,
        include_label=False,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=32, collate_fn=list_data_collate, num_workers=0
    )
    model = load_model().to(DEVICE)
    probs = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(DEVICE)  # now a Tensor [B,C,H,W]
            logits = model(imgs)
            p = F.softmax(logits, dim=1)[:, 1]
            probs.append(p.cpu())
    mean_p = torch.cat(probs).mean().item()
    label = "malignant" if mean_p >= 0.5 else "benign"
    return f"Mean slide: {label} ({mean_p:.4f})"
