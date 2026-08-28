import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.evaluation.segmentation import build_site_lookup, day_night_label


def test_day_night_label_night_for_grayscale_image(tmp_path):
    path = tmp_path / "night.jpg"
    arr = np.full((32, 32, 3), 128, dtype="uint8")
    Image.fromarray(arr).save(path)

    assert day_night_label(path) == "night"


def test_day_night_label_day_for_color_image(tmp_path):
    path = tmp_path / "day.jpg"
    arr = np.zeros((32, 32, 3), dtype="uint8")
    arr[..., 0] = 200
    Image.fromarray(arr).save(path)

    assert day_night_label(path) == "day"


def test_build_site_lookup_maps_filename_to_location():
    images_df = pd.DataFrame({"file_name": ["a.jpg", "b.jpg"], "location": ["12", "45"]})

    lookup = build_site_lookup(images_df)

    assert lookup == {"a.jpg": "12", "b.jpg": "45"}
