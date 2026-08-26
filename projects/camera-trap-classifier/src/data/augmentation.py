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
