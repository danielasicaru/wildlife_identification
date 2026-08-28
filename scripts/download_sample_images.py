"""Downloads a stratified sample of real CCT images (not the full 105GB archive).

Samples up to N images per species (from the annotation labels) plus a share of `empty` frames,
so downstream image-statistics and quality-check work has real, representative pixel data without
pulling the whole dataset.
"""
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_annotations, merge_categories
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
IMAGE_BASE_URL = "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images"
ANNOTATIONS_PATH = ROOT / "data" / "raw" / "caltech_images_20210113.json"
IMAGES_DIR = ROOT / "data" / "raw" / "images"

config = load_config(ROOT / "configs" / "download_sample_images.yaml")
PER_CLASS_SAMPLE_SIZE = config["per_class_sample_size"]
SEED = config["seed"]


def main() -> None:
    images, annotations, categories = load_annotations(ANNOTATIONS_PATH)

    merged = merge_categories(annotations, categories)
    merged = merged.merge(images[["id", "file_name"]], left_on="image_id", right_on="id", suffixes=("", "_img"))
    # An image with multiple animals of the same species has multiple annotation rows here; drop
    # to one row per (species, file) so sampling can't pick the same file twice within a category.
    merged = merged.drop_duplicates(subset=["name", "file_name"])

    sample_parts = []
    for _, group in merged.groupby("name"):
        sample_parts.append(group.sample(n=min(PER_CLASS_SAMPLE_SIZE, len(group)), random_state=SEED))
    sample = pd.concat(sample_parts, ignore_index=True)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(sample)} images across {sample['name'].nunique()} categories...")

    for i, row in enumerate(sample.itertuples(), start=1):
        dest = IMAGES_DIR / row.file_name
        if dest.exists():
            continue
        url = f"{IMAGE_BASE_URL}/{row.file_name}"
        urllib.request.urlretrieve(url, dest)
        if i % 50 == 0:
            print(f"  {i}/{len(sample)} downloaded")

    print(f"Done. Images in {IMAGES_DIR}")


if __name__ == "__main__":
    main()
