import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from monai.networks.nets import EfficientNetBN
from monai.visualize.class_activation_maps import ModelWithHooks
from monai.visualize.gradient_based import SmoothGrad
from captum.attr import LayerGradCam

# Device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)
MODEL_PATH = "../models/tissue-classification/efficientnet-b3.pth"
CLASSES = ["benign", "malignant"]

# Initialize and load base model
base_model = EfficientNetBN(
    model_name="efficientnet-b3",
    pretrained=False,
    spatial_dims=2,
    in_channels=3,
    num_classes=len(CLASSES)
).to(device)
base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
base_model.eval()

# Wrap model for Grad-CAM and SmoothGrad hooks
model_with_hooks = ModelWithHooks(
    base_model,
    target_layer_names=("_conv_head",),
    register_forward=True,
    register_backward=True
)

# Preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Generate Grad-CAM overlay
def generate_grad_cam_overlay(pil_img: Image.Image, wrapper: ModelWithHooks, class_idx: int, alpha: float = 0.5):
    img = pil_img.convert('RGB')
    inp = preprocess(img).unsqueeze(0).to(device)

    # Compute Grad-CAM at target layer
    base = wrapper.model
    target_layer = wrapper.get_layer("_conv_head")
    gradcam = LayerGradCam(base, target_layer)
    attributions = gradcam.attribute(inp, target=class_idx)

    # Process attribution
    attr = attributions.squeeze(0)
    cam = torch.clamp(attr, min=0).sum(dim=0)
    cam -= cam.min()
    cam /= cam.max() + 1e-7

    # Upsample and smooth
    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(img.height, img.width),
        mode='bilinear',
        align_corners=False
    )[0, 0]
    cam_np = cam_up.detach().cpu().numpy()
    cam_np = cv2.GaussianBlur(cam_np, (5, 5), 0)

    # Create overlay
    heat = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(img), 1 - alpha, heat, alpha, 0)
    return Image.fromarray(overlay)


# Generate SmoothGrad overlay using MONAI's SmoothGrad

def generate_smoothgrad_overlay(pil_img: Image.Image, wrapper: ModelWithHooks, class_idx: int, alpha: float = 0.6):
    img = pil_img.convert('RGB')
    inp = preprocess(img).unsqueeze(0).to(device)

    # Use MONAI's SmoothGrad
    smooth = SmoothGrad(
        wrapper,
        n_samples=25,
        magnitude=True,
        stdev_spread=0.04,
        verbose=False
    )
    sal = smooth(inp, index=class_idx)  # returns gradients tensor

    # Collapse channels and normalize to 2D
    sal_np = sal.squeeze(0).cpu().numpy()  # CxHxW
    sal_map = np.mean(np.abs(sal_np), axis=0)
    sal_map -= sal_map.min()
    sal_map /= sal_map.max() + 1e-7

    # Upsample & smooth
    sal_up = cv2.resize(sal_map, (img.width, img.height), interpolation=cv2.INTER_CUBIC)
    sal_up = cv2.GaussianBlur(sal_up, (5, 5), 0)

    # Create red heatmap
    heat_np = np.zeros((img.height, img.width, 3), dtype=np.uint8)
    heat_np[..., 0] = (sal_up * 255).astype(np.uint8)

    # Blend with original
    overlay = cv2.addWeighted(np.array(img), 1 - alpha, heat_np, alpha, 0)
    return Image.fromarray(overlay)


# Inference + explanation
def predict_and_explain(filepaths):
    # Load and preprocess all images
    images = [Image.open(fp).convert('RGB') for fp in filepaths]
    batch_tensors = [preprocess(img) for img in images]
    inp_batch = torch.stack(batch_tensors, dim=0).to(device)

    # Forward pass
    with torch.no_grad():
        logits = base_model(inp_batch)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        confs = probs[range(len(images)), preds]

    predictions = []
    cam_overlays = []
    smooth_overlays = []

    # Per-image explainability
    for idx_img, img in enumerate(images):
        class_idx = int(preds[idx_img].item())
        label = f"{CLASSES[class_idx]} ({float(confs[idx_img]):.2f})"
        predictions.append(label)

        # Setup fresh hooks wrapper per image
        wrapper = ModelWithHooks(
            base_model,
            target_layer_names=("_conv_head",),
            register_forward=True,
            register_backward=True
        )
        cam = generate_grad_cam_overlay(img, wrapper, class_idx=class_idx)
        smooth = generate_smoothgrad_overlay(img, wrapper, class_idx=class_idx)
        wrapper.gradients.clear()
        wrapper.activations.clear()

        cam_overlays.append(cam)
        smooth_overlays.append(smooth)

    return "\n".join(predictions), cam_overlays, smooth_overlays


# Gradio interface
inputs = gr.Files(file_count="multiple", type="filepath", label="Input Images")
outputs = [
    gr.Textbox(label="Predictions", lines=5),
    gr.Gallery(label="Grad-CAM Overlays", columns=2, height="auto"),
    gr.Gallery(label="SmoothGrad Overlays", columns=2, height="auto")
]

demo = gr.Interface(
    fn=predict_and_explain,
    inputs=inputs,
    outputs=outputs,
    title="EfficientNet Batch Classifier with Grad-CAM & SmoothGrad",
    description=(
        "Upload one or more images to get batch predictions plus individual Grad-CAM and "
        "SmoothGrad overlays using MONAI’s implementation."
    )
)

if __name__ == "__main__":
    demo.launch()
