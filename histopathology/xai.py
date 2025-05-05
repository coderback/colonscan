# xai.py

import torch
import torch.nn as nn
import torchvision.models as models
from captum.attr import LayerGradCam
import shap
import numpy as np
import cv2
from PIL import Image

# 1) Model factory
def get_model(num_classes):
    """
    Return a ResNet50 with its final FC layer replaced for `num_classes`.
    """
    model = models.resnet50(pretrained=True)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

# 2) Mapping from label name -> class index
#    Adjust these keys to match your two folders, e.g. "tumor" / "non_tumor"
class_to_idx = {
    'non_tumor': 0,
    'tumor':      1,
}


def compute_saliency_map(model, input_tensor, target_class):
    """
    Vanilla saliency: max over abs gradients of input.
    Returns a PIL.Image of the overlaid heatmap.
    """
    model.zero_grad()
    inp = input_tensor.clone().detach().requires_grad_(True)
    logits = model(inp)
    score = logits[0, target_class]
    score.backward()

    saliency, _ = torch.max(inp.grad.abs(), dim=1)         # (1,H,W)
    saliency = saliency.squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = np.uint8(255 * saliency)

    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)  # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # reconstruct original image
    orig = input_tensor.squeeze().cpu().permute(1,2,0).numpy()
    orig = orig * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
    orig = np.clip(orig,0,1)
    orig = np.uint8(255 * orig)

    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlay)


def compute_gradcam_map(model, input_tensor, target_class):
    """
    Grad-CAM via Captum on the last ResNet block.
    Returns a PIL.Image of the overlaid heatmap.
    """
    # pick the last conv layer of ResNet-50
    target_layer = model.layer4[-1].conv3

    gradcam = LayerGradCam(model, target_layer)
    attributions = gradcam.attribute(input_tensor, target=target_class)
    heat = attributions.squeeze().cpu().detach().numpy()
    heat = np.maximum(heat, 0)
    heat = heat / (heat.max() + 1e-8)
    heat = np.uint8(255 * heat)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    orig = input_tensor.squeeze().cpu().permute(1,2,0).numpy()
    orig = orig * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
    orig = np.clip(orig,0,1)
    orig = np.uint8(255 * orig)

    overlay = cv2.addWeighted(orig, 0.6, heat, 0.4, 0)
    return Image.fromarray(overlay)


def compute_shap_map(model, input_tensor, target_class):
    """
    SHAP via a gradient explainer.  Returns a PIL.Image of the overlaid map.
    """
    # use a zero tensor as background
    background = torch.zeros_like(input_tensor)
    explainer = shap.GradientExplainer(model, background)

    # shap_values is a list over classes
    shap_vals = explainer.shap_values(input_tensor)
    # pick the chosen class, then batch‐0
    arr = shap_vals[target_class][0]   # shape (3,H,W)

    # aggregate over channels
    heat = np.sum(np.abs(arr), axis=0)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat = np.uint8(255 * heat)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    orig = input_tensor.squeeze().cpu().permute(1,2,0).numpy()
    orig = orig * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
    orig = np.clip(orig,0,1)
    orig = np.uint8(255 * orig)

    overlay = cv2.addWeighted(orig, 0.6, heat, 0.4, 0)
    return Image.fromarray(overlay)
