from unittest.mock import MagicMock

import torch
from PIL import Image

from src.api.inference import predict


class FakeClassifier:
    """Returns a fixed logit vector regardless of input -- lets tests assert on the label
    mapping/response shape without running a real forward pass."""

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, x):
        return self._logits.unsqueeze(0)

    def eval(self):
        return self


def _fake_state(detections, logits, species_to_index):
    detector = MagicMock()
    detector_module_result = {"detections": detections}

    state = MagicMock()
    state.detector = detector
    state.classifier = FakeClassifier(logits)
    state.species_to_index = species_to_index
    state.index_to_species = {v: k for k, v in species_to_index.items()}
    state.device = "cpu"
    state.min_confidence = 0.2
    state.box_expansion_fraction = 0.1

    from src.data.augmentation import build_val_transform
    state.val_transform = build_val_transform()

    return state, detector_module_result


def test_predict_returns_empty_list_when_no_animal_detected(monkeypatch):
    species_to_index = {"fox": 0, "coyote": 1}
    state, raw_result = _fake_state([], torch.tensor([0.0, 0.0]), species_to_index)

    monkeypatch.setattr("src.api.inference.run_detection", lambda *a, **k: raw_result)

    image = Image.new("RGB", (100, 100))
    result = predict(image, state)

    assert result == []


def test_predict_returns_bbox_species_confidence_per_detection(monkeypatch):
    species_to_index = {"fox": 0, "coyote": 1}
    detections = [{"category": "1", "conf": 0.9, "bbox": [0.1, 0.1, 0.5, 0.5]}]
    logits = torch.tensor([5.0, 0.0])  # heavily favors "fox" (index 0)
    state, raw_result = _fake_state(detections, logits, species_to_index)

    monkeypatch.setattr("src.api.inference.run_detection", lambda *a, **k: raw_result)

    image = Image.new("RGB", (100, 100))
    result = predict(image, state)

    assert len(result) == 1
    assert result[0]["species"] == "fox"
    assert result[0]["confidence"] > 0.9
    assert len(result[0]["bbox"]) == 4
