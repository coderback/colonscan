import torch
import torch.nn.functional as F
import gradio as gr
from monai.data import WSIReader, SlidingPatchWSIDataset, list_data_collate
from monai.networks.nets import EfficientNetBN
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    ResizeD,
    ScaleIntensityRangeD,
    ToTensord,
)

# Global settings
MODEL_PATH = "../models/tissue-classification/efficientnet-b3.pth"
device = torch.device("cuda")
CLASSES = ["benign", "malignant"]
IMG_SIZE = (224, 224)


def load_model():
    if not hasattr(load_model, "model"):
        # Initialize model architecture
        model = EfficientNetBN(
            spatial_dims=2,
            in_channels=3,
            model_name="efficientnet-b3",
            pretrained=True,
            num_classes=2,
        ).to(device)
        try:
            state = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state)
        except Exception as e:
            raise RuntimeError(f"Failed to load model checkpoint '{MODEL_PATH}': {e}")
        model.eval()
        load_model.model = model
    return load_model.model


def get_patch_transforms(patch_size):
    return Compose([
        EnsureChannelFirstD(keys="image"),
        ResizeD(keys="image", spatial_size=(patch_size, patch_size), mode="bilinear"),
        ScaleIntensityRangeD(
            keys="image",
            a_min=0, a_max=255,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        ToTensord(keys="image"),
    ])


# WSI inference
def infer_wsi(
        slide_file,
        patch_size: int = 224,
        overlap: float = 0.5
):
    # Prepare dataset and loader
    slide_path = slide_file.name
    reader = WSIReader("openslide")
    dataset = SlidingPatchWSIDataset(
        data=[{"image": slide_path}],
        patch_size=patch_size,
        overlap=overlap,
        transform=get_patch_transforms(patch_size),
        reader=reader,
        center_location=False,
        include_label=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        collate_fn=list_data_collate,
        num_workers=0,
    )

    # Aggregate probabilities
    model = load_model()
    probs = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            logits = model(imgs)
            p = F.softmax(logits, dim=1)[:, 1]
            probs.append(p.cpu())
    mean_p = torch.cat(probs).mean().item()
    label = "malignant" if mean_p >= 0.5 else "benign"
    return f"Mean slide: {label} ({mean_p:.4f})"


# Gradio app
if __name__ == "__main__":
    iface = gr.Interface(
        fn=infer_wsi,
        inputs=[
            gr.File(label="Whole‐Slide Image (.svs, etc.)"),
            gr.Slider(64, 512, value=224, step=32, label="Patch size"),
            gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Patch overlap"),
        ],
        outputs="text",
        title="WSI Inference with EfficientNet-b3 (2-class)",
        description=(
            "Upload a whole-slide image to compute the mean probability of class 1 "
            "using a hardcoded EfficientNet-b3 checkpoint on CUDA."
        ),
        flagging_mode="never",
    )
    iface.launch()
