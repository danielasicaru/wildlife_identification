import pytest
import torch

from src.classifier.models import build_model
from src.evaluation.gradcam import grad_cam


@pytest.mark.parametrize("backbone", ["resnet50", "efficientnet_b0"])
def test_grad_cam_returns_heatmap_matching_input_spatial_size(backbone):
    model = build_model(backbone, num_classes=5, pretrained=False)
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)

    heatmap = grad_cam(model, backbone, input_tensor, target_class=0)

    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_grad_cam_raises_for_unsupported_backbone():
    model = build_model("vit_b_16", num_classes=5, pretrained=False)
    input_tensor = torch.randn(1, 3, 224, 224)

    with pytest.raises(ValueError):
        grad_cam(model, "vit_b_16", input_tensor, target_class=0)
