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


from torchvision.transforms import v2

from src.data.augmentation import build_train_transform


def test_train_transform_produces_correct_shape_and_dtype():
    for is_minority in (True, False):
        transform = build_train_transform(is_minority=is_minority)
        img = _sample_image()

        result = transform(img)

        assert result.shape == (3, 224, 224)
        assert result.dtype == torch.float32


def _find_random_apply(steps, wrapped_type):
    for step in steps:
        if isinstance(step, v2.RandomApply):
            if isinstance(step.transforms[0], wrapped_type):
                return step
    return None


def test_minority_crop_is_always_applied_not_wrapped():
    transform = build_train_transform(is_minority=True)

    crop_steps = [s for s in transform.transforms if isinstance(s, v2.RandomResizedCrop)]
    wrapped_crop = _find_random_apply(transform.transforms, v2.RandomResizedCrop)

    assert len(crop_steps) == 1  # applied directly, p=1.0, no RandomApply wrapper needed
    assert wrapped_crop is None


def test_majority_crop_is_wrapped_with_probability_point_eight():
    transform = build_train_transform(is_minority=False)

    wrapped_crop = _find_random_apply(transform.transforms, v2.RandomResizedCrop)

    assert wrapped_crop is not None
    assert wrapped_crop.p == pytest.approx(0.8)


def test_erasing_probability_differs_majority_minority():
    majority = build_train_transform(is_minority=False)
    minority = build_train_transform(is_minority=True)

    majority_erasing = next(s for s in majority.transforms if isinstance(s, v2.RandomErasing))
    minority_erasing = next(s for s in minority.transforms if isinstance(s, v2.RandomErasing))

    assert majority_erasing.p == pytest.approx(0.5)
    assert minority_erasing.p == pytest.approx(0.3)


def test_hue_saturation_jitter_only_present_for_majority():
    majority = build_train_transform(is_minority=False)
    minority = build_train_transform(is_minority=True)

    majority_hue_jitters = [
        s for s in majority.transforms
        if isinstance(s, v2.RandomApply) and isinstance(s.transforms[0], v2.ColorJitter)
        and s.transforms[0].hue is not None
    ]
    minority_hue_jitters = [
        s for s in minority.transforms
        if isinstance(s, v2.RandomApply) and isinstance(s.transforms[0], v2.ColorJitter)
        and s.transforms[0].hue is not None
    ]

    assert len(majority_hue_jitters) == 1
    assert majority_hue_jitters[0].p == pytest.approx(0.3)
    assert minority_hue_jitters == []


def test_gaussian_noise_only_present_for_majority():
    majority = build_train_transform(is_minority=False)
    minority = build_train_transform(is_minority=True)

    majority_noise = _find_random_apply(majority.transforms, v2.GaussianNoise)
    minority_noise = _find_random_apply(minority.transforms, v2.GaussianNoise)

    assert majority_noise is not None
    assert majority_noise.p == pytest.approx(0.2)
    assert minority_noise is None
