import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from monai.networks.nets import EfficientNetBN
from monai.transforms import Compose, EnsureChannelFirstD, ResizeD, ScaleIntensityRangeD, ToTensord
from monai.visualize.class_activation_maps import ModelWithHooks
from monai.visualize.gradient_based import SmoothGrad
from captum.attr import LayerGradCam
from monai.data import WSIReader, SlidingPatchWSIDataset, list_data_collate

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
    return Compose([
        EnsureChannelFirstD(keys="image"),
        ResizeD(keys="image", spatial_size=(patch_size, patch_size), mode="bilinear"),
        ScaleIntensityRangeD(keys="image", a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys="image"),
    ])

# ---- Utility ----
def to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ---- Grad-CAM ----
def compute_gradcam_map(
    pil_img: Image.Image,
    class_idx: int,
    patch_size: int = 224,
    alpha: float = 0.5
) -> Image.Image:
    model = load_model()
    wrapper = ModelWithHooks(
        model,
        target_layer_names=("_conv_head",),
        register_forward=True,
        register_backward=True
    )
    # preprocess
    img = pil_img.convert("RGB")
    x = get_patch_transforms(patch_size)({"image": np.array(img)})["image"].unsqueeze(0).to(DEVICE)
    # attribute
    gradcam = LayerGradCam(model, wrapper.get_layer("_conv_head"))
    attributions = gradcam.attribute(x, target=class_idx)
    # build heatmap
    attr = attributions.squeeze(0)
    cam = torch.clamp(attr, min=0).sum(dim=0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(img.height, img.width),
        mode="bilinear", align_corners=False
    )[0,0].cpu().numpy()
    cam_up = cv2.GaussianBlur(cam_up, (5,5), 0)
    heat = cv2.applyColorMap((cam_up*255).astype(np.uint8), cv2.COLORMAP_JET)
    over = cv2.addWeighted(np.array(img), 1-alpha, heat, alpha, 0)
    wrapper.gradients.clear(); wrapper.activations.clear()
    return Image.fromarray(over)

# ---- SmoothGrad (“saliency”) ----
def compute_saliency_map(
    pil_img: Image.Image,
    class_idx: int,
    patch_size: int = 224,
    alpha: float = 0.6
) -> Image.Image:
    model = load_model()
    wrapper = ModelWithHooks(
        model,
        target_layer_names=("_conv_head",),
        register_forward=True,
        register_backward=True
    )
    img = pil_img.convert("RGB")
    x = get_patch_transforms(patch_size)({"image": np.array(img)})["image"].unsqueeze(0).to(DEVICE)
    # SmoothGrad
    smooth = SmoothGrad(
        wrapper,
        n_samples=25,
        magnitude=True,
        stdev_spread=0.04,
        verbose=False
    )
    sal = smooth(x, index=class_idx).squeeze(0).cpu().numpy()  # C×H×W
    sal_map = np.mean(np.abs(sal), axis=0)
    sal_map = (sal_map - sal_map.min())/(sal_map.max()-sal_map.min()+1e-7)
    sal_up = cv2.resize(sal_map, (img.width, img.height), interpolation=cv2.INTER_CUBIC)
    sal_up = cv2.GaussianBlur(sal_up, (5,5), 0)
    # red channel heatmap
    heat = np.zeros((img.height, img.width, 3), np.uint8)
    heat[...,0] = (sal_up*255).astype(np.uint8)
    over = cv2.addWeighted(np.array(img), 1-alpha, heat, alpha, 0)
    wrapper.gradients.clear(); wrapper.activations.clear()
    return Image.fromarray(over)

# ---- WSI inference ----
def infer_wsi(slide_path: str, patch_size: int=224, overlap: float=0.5) -> str:
    reader = WSIReader("openslide")
    ds = SlidingPatchWSIDataset(
        data=[{"image": slide_path}],
        patch_size=patch_size,
        overlap=overlap,
        transform=get_patch_transforms(patch_size),
        reader=reader,
        center_location=False,
        include_label=False
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=32, collate_fn=list_data_collate, num_workers=0
    )
    model = load_model()
    probs = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(DEVICE)
            logits = model(imgs)
            p = F.softmax(logits, dim=1)[:,1]
            probs.append(p.cpu())
    mean_p = torch.cat(probs).mean().item()
    label = "malignant" if mean_p>=0.5 else "benign"
    return f"Mean slide: {label} ({mean_p:.4f})"
