"""Training/validation augmentation pipelines, per the augmentation design spec section 2."""
import pandas as pd
import torch
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_INPUT_SIZE = 224


def minority_species(counts: pd.Series) -> set[str]:
    """Species names with annotation count strictly below the median count across all species."""
    median = counts.median()
    return set(counts[counts < median].index)


def build_sample_weights(labels: pd.Series, counts: pd.Series) -> list[float]:
    """Per-sample inverse-class-frequency weight, for a WeightedRandomSampler."""
    return [1.0 / counts[label] for label in labels]


def build_val_transform() -> v2.Compose:
    """No augmentation: resize to model input and normalize, identical every call."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_train_transform(is_minority: bool) -> v2.Compose:
    """Training augmentation pipeline with differential per-class probabilities.

    Probabilities and parameters are pinned to the augmentation design spec's training pipeline
    table. GaussianNoise's sigma is a fixed midpoint (0.02) representing the spec's 0.01-0.03
    range, since GaussianNoise doesn't support a sigma range natively.
    """
    steps = [
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.RandomRotation(degrees=15)], p=0.3),
    ]

    crop = v2.RandomResizedCrop(MODEL_INPUT_SIZE, scale=(0.4, 1.0), ratio=(0.9, 1.1))
    if is_minority:
        steps.append(crop)  # p=1.0 per spec: always applied, no wrapper needed
    else:
        steps.append(v2.RandomApply([crop], p=0.8))

    erasing_p = 0.3 if is_minority else 0.5
    steps.append(v2.RandomErasing(p=erasing_p, scale=(0.1, 0.2)))

    brightness_jitter = (
        v2.ColorJitter(brightness=0.1) if is_minority
        else v2.ColorJitter(brightness=0.2, contrast=0.15)
    )
    steps.append(v2.RandomApply([brightness_jitter], p=0.4))

    grayscale_p = 0.05 if is_minority else 0.15
    steps.append(v2.RandomGrayscale(p=grayscale_p))

    if not is_minority:
        steps.append(v2.RandomApply([v2.ColorJitter(hue=0.05, saturation=0.1)], p=0.3))

    steps += [
        # No-op when the crop step above fired (it already outputs MODEL_INPUT_SIZE), but
        # required when it didn't (majority images that skip the p=0.8 crop still need resizing).
        v2.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
        v2.ToDtype(torch.float32, scale=True),
    ]

    if not is_minority:
        steps.append(v2.RandomApply([v2.GaussianNoise(sigma=0.02)], p=0.2))

    steps.append(v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return v2.Compose(steps)
