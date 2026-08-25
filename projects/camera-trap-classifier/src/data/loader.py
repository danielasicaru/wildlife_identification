"""Loads COCO-Camera-Traps-format annotation JSON into pandas DataFrames."""
import json
from pathlib import Path

import pandas as pd


def load_annotations(json_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load images, annotations, and categories tables from a CCT-format JSON file.

    Returns (images_df, annotations_df, categories_df).
    """
    with open(json_path) as f:
        data = json.load(f)

    images = pd.DataFrame(data["images"])
    annotations = pd.DataFrame(data["annotations"])
    categories = pd.DataFrame(data["categories"])[["id", "name"]]

    return images, annotations, categories
