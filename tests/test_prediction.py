import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from src.classifier.prediction import predict_test_set


class FakeClassifier(nn.Module):
    """Ignores its input and always predicts index 1 with high confidence."""

    def forward(self, x):
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, 2)
        logits[:, 1] = 10.0
        return logits


def test_predict_test_set_returns_one_row_per_crop(tmp_path):
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    for name in ["a.jpg", "b.jpg"]:
        Image.new("RGB", (32, 32), color=(100, 100, 100)).save(crops_dir / name)

    test_df = pd.DataFrame({
        "crop_file": ["a.jpg", "b.jpg"],
        "source_image": ["src_a.jpg", "src_b.jpg"],
        "species": ["fox", "coyote"],
    })
    species_to_index = {"fox": 0, "coyote": 1}
    index_to_species = {0: "fox", 1: "coyote"}

    result = predict_test_set(FakeClassifier(), test_df, crops_dir, species_to_index, index_to_species, device="cpu")

    assert list(result.columns) == ["crop_file", "true", "predicted", "confidence"]
    assert len(result) == 2
    assert list(result["crop_file"]) == ["a.jpg", "b.jpg"]
    assert list(result["true"]) == ["fox", "coyote"]
    assert (result["predicted"] == "coyote").all()
    assert (result["confidence"] > 0.9).all()
