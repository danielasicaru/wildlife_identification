import json
import pandas as pd

from src.data.loader import load_annotations, merge_categories

FIXTURE = {
    "images": [
        {"id": "img1", "file_name": "img1.jpg", "location": "26",
         "date_captured": "2013-10-04 13:31:53", "height": 1494, "width": 2048},
        {"id": "img2", "file_name": "img2.jpg", "location": "26",
         "date_captured": "2013-10-04 02:10:00", "height": 1494, "width": 2048},
    ],
    "annotations": [
        {"id": "a1", "image_id": "img1", "category_id": 6},
        {"id": "a2", "image_id": "img2", "category_id": 30},
    ],
    "categories": [
        {"id": 6, "name": "bobcat"},
        {"id": 30, "name": "empty"},
    ],
    "info": {"version": "test"},
}


def test_load_annotations_returns_three_dataframes(tmp_path):
    fixture_path = tmp_path / "annotations.json"
    fixture_path.write_text(json.dumps(FIXTURE))

    images, annotations, categories = load_annotations(fixture_path)

    assert isinstance(images, pd.DataFrame)
    assert isinstance(annotations, pd.DataFrame)
    assert isinstance(categories, pd.DataFrame)
    assert len(images) == 2
    assert len(annotations) == 2
    assert len(categories) == 2
    assert list(categories.columns) == ["id", "name"]


def test_load_annotations_image_ids_are_unique(tmp_path):
    fixture_path = tmp_path / "annotations.json"
    fixture_path.write_text(json.dumps(FIXTURE))

    images, _, _ = load_annotations(fixture_path)

    assert images["id"].is_unique


def test_merge_categories_adds_species_name_column(tmp_path):
    fixture_path = tmp_path / "annotations.json"
    fixture_path.write_text(json.dumps(FIXTURE))
    _, annotations, categories = load_annotations(fixture_path)

    merged = merge_categories(annotations, categories)

    assert len(merged) == len(annotations)
    assert set(merged["name"]) == {"bobcat", "empty"}
