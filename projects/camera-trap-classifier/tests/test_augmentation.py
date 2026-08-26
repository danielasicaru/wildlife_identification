import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.augmentation import build_sample_weights, build_val_transform, minority_species


def _sample_image(width=640, height=480):
    arr = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype="uint8")
    return Image.fromarray(arr)


def test_minority_species_below_median_count():
    counts = pd.Series({"bobcat": 100, "coyote": 50, "fox": 10, "badger": 5}, name="count")

    result = minority_species(counts)

    assert result == {"fox", "badger"}


def test_minority_species_empty_when_all_equal():
    counts = pd.Series({"bobcat": 10, "coyote": 10}, name="count")

    result = minority_species(counts)

    assert result == set()


def test_build_sample_weights_inverse_frequency():
    counts = pd.Series({"bobcat": 100, "fox": 10})
    labels = pd.Series(["bobcat", "bobcat", "fox"])

    weights = build_sample_weights(labels, counts)

    assert weights[0] == pytest.approx(1.0 / 100)
    assert weights[2] == pytest.approx(1.0 / 10)
    assert weights[2] > weights[0]


def test_val_transform_produces_correct_shape_and_dtype():
    transform = build_val_transform()
    img = _sample_image()

    result = transform(img)

    assert result.shape == (3, 224, 224)
    assert result.dtype == torch.float32


def test_val_transform_is_deterministic():
    transform = build_val_transform()
    img = _sample_image()

    result_a = transform(img)
    result_b = transform(img)

    assert torch.allclose(result_a, result_b)
