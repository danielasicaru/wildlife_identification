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
