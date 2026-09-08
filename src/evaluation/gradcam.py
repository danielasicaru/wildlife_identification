"""Grad-CAM: visualizes which regions of a crop most influenced the classifier's prediction, by
weighting the last convolutional layer's feature maps by the gradient of the predicted class
score. CNN-only (ResNet50, EfficientNet-B0) -- ViT-B/16 has no equivalent spatial conv feature
map to hook into, so it isn't supported here."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _last_conv_layer(model: nn.Module, backbone: str) -> nn.Module:
    if backbone == "resnet50":
        return model.layer4[-1]
    if backbone == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Grad-CAM isn't supported for backbone {backbone!r} (no spatial conv feature map).")


def grad_cam(model: nn.Module, backbone: str, input_tensor: torch.Tensor, target_class: int) -> torch.Tensor:
    """input_tensor: a single preprocessed image, shape (1, C, H, W). Returns an (H, W) heatmap
    normalized to [0, 1] (all-zero if the target class's score doesn't vary spatially at all),
    upsampled to the same spatial size as input_tensor.
    """
    layer = _last_conv_layer(model, backbone)
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(module, args, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_forward = layer.register_forward_hook(forward_hook)
    handle_backward = layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()
        output = model(input_tensor)
        output[0, target_class].backward()

        activation = activations["value"][0]  # (C, h, w)
        gradient = gradients["value"][0]  # (C, h, w)
        weights = gradient.mean(dim=(1, 2))  # (C,) -- global-average-pooled gradient per channel

        cam = torch.relu((weights[:, None, None] * activation).sum(dim=0))  # (h, w)
        cam = F.interpolate(
            cam[None, None], size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        return cam.detach()
    finally:
        handle_forward.remove()
        handle_backward.remove()
