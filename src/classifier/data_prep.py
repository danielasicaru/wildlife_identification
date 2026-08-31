"""Builds the labeled, species-count-filtered crop dataframe shared by every classifier
training/evaluation script, so each one reconstructs identical crops from the same
detections/annotations/bbox files instead of re-deriving the join/filter logic separately."""
import json
from pathlib import Path

import pandas as pd

from src.classifier.labeling import build_crop_dataframe
from src.data.loader import load_annotations, merge_categories

NON_SPECIES = {"empty", "car"}


def build_labeled_crop_df(
    detections_path: Path, annotations_path: Path, bbox_path: Path, min_samples_per_species: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (crop_df, images_df). crop_df has columns [crop_file, source_image, species],
    filtered to species with at least min_samples_per_species crops dataset-wide. images_df is the
    raw CCT images table (needed by callers for fields like `location`).
    """
    with open(detections_path, encoding="utf-8") as f:
        detections = json.load(f)

    images_df, annotations_df, categories_df = load_annotations(annotations_path)
    merged = merge_categories(annotations_df, categories_df)
    merged = merged.merge(images_df[["id", "file_name"]], left_on="image_id", right_on="id", suffixes=("", "_img"))
    merged = merged[~merged["name"].isin(NON_SPECIES)]
    image_species = merged.groupby("file_name")["name"].apply(set).to_dict()

    with open(bbox_path, encoding="utf-8") as f:
        bbox_data = json.load(f)
    bbox_images = {im["id"]: im["file_name"] for im in bbox_data["images"]}
    image_ground_truth: dict[str, list] = {}
    for ann in bbox_data["annotations"]:
        file_name = bbox_images.get(ann["image_id"])
        species = image_species.get(file_name)
        if file_name is None or not species or len(species) != 1:
            continue
        bbox_abs = tuple(round(v) for v in ann["bbox"])
        image_ground_truth.setdefault(file_name, []).append((bbox_abs, next(iter(species))))

    crop_df = build_crop_dataframe(detections, image_species, image_ground_truth)
    species_counts = crop_df["species"].value_counts()
    crop_df = crop_df[crop_df["species"].map(species_counts) >= min_samples_per_species].reset_index(drop=True)

    return crop_df, images_df
