"""Backbone factory for the model comparison: ResNet50, EfficientNet-B0, ViT-B/16, all
transfer-learned with a replaced classification head."""
import torch.nn as nn
import torchvision.models as models

BACKBONES = ("resnet50", "efficientnet_b0", "vit_b_16")


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}. Expected one of {BACKBONES}.")

    return model
