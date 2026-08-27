import pytest
import torch

from src.classifier.models import BACKBONES, build_model


@pytest.mark.parametrize("backbone", BACKBONES)
def test_build_model_output_matches_num_classes(backbone):
    model = build_model(backbone, num_classes=5, pretrained=False)
    x = torch.randn(2, 3, 224, 224)

    output = model(x)

    assert output.shape == (2, 5)


def test_build_model_unknown_backbone_raises():
    with pytest.raises(ValueError):
        build_model("not_a_real_backbone", num_classes=5, pretrained=False)
