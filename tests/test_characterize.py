import pandas as pd
import pytest

from src.data.characterize import (
    aspect_ratio_survey,
    bbox_area_ratio,
    class_distribution,
    metadata_survey,
    per_site_distribution,
)

SPECIES_ONLY_EXCLUDES = {"empty", "car"}


def _sample_tables():
    annotations = pd.DataFrame([
        {"id": "a1", "image_id": "img1", "category_id": 6},
        {"id": "a2", "image_id": "img2", "category_id": 6},
        {"id": "a3", "image_id": "img3", "category_id": 9},
        {"id": "a4", "image_id": "img4", "category_id": 30},  # empty, excluded
    ])
    categories = pd.DataFrame([
        {"id": 6, "name": "bobcat"},
        {"id": 9, "name": "coyote"},
        {"id": 30, "name": "empty"},
    ])
    return annotations, categories


def test_class_distribution_counts_per_species():
    annotations, categories = _sample_tables()

    result = class_distribution(annotations, categories, exclude=SPECIES_ONLY_EXCLUDES)

    assert result.loc["bobcat", "count"] == 2
    assert result.loc["coyote", "count"] == 1


def test_class_distribution_excludes_non_species_categories():
    annotations, categories = _sample_tables()

    result = class_distribution(annotations, categories, exclude=SPECIES_ONLY_EXCLUDES)

    assert "empty" not in result.index


def test_class_distribution_includes_imbalance_ratio():
    annotations, categories = _sample_tables()

    result = class_distribution(annotations, categories, exclude=SPECIES_ONLY_EXCLUDES)

    assert "imbalance_ratio" in result.columns
    max_count = result["count"].max()
    min_count = result["count"].min()
    assert result.loc[result["count"].idxmax(), "imbalance_ratio"] == max_count / min_count


def _sample_images_and_annotations():
    images = pd.DataFrame([
        {"id": "img1", "location": "26"},
        {"id": "img2", "location": "26"},
        {"id": "img3", "location": "40"},
        {"id": "img4", "location": "40"},
    ])
    annotations = pd.DataFrame([
        {"id": "a1", "image_id": "img1", "category_id": 6},
        {"id": "a2", "image_id": "img2", "category_id": 6},
        {"id": "a3", "image_id": "img3", "category_id": 9},
        {"id": "a4", "image_id": "img4", "category_id": 30},
    ])
    categories = pd.DataFrame([
        {"id": 6, "name": "bobcat"},
        {"id": 9, "name": "coyote"},
        {"id": 30, "name": "empty"},
    ])
    return images, annotations, categories


def test_per_site_distribution_pivots_species_by_location():
    images, annotations, categories = _sample_images_and_annotations()

    result = per_site_distribution(images, annotations, categories, exclude=SPECIES_ONLY_EXCLUDES)

    assert result.loc["26", "bobcat"] == 2
    assert result.loc["40", "coyote"] == 1


def test_per_site_distribution_excludes_non_species_categories():
    images, annotations, categories = _sample_images_and_annotations()

    result = per_site_distribution(images, annotations, categories, exclude=SPECIES_ONLY_EXCLUDES)

    assert "empty" not in result.columns


def test_metadata_survey_reports_resolution_counts():
    images = pd.DataFrame([
        {"id": "img1", "height": 1494, "width": 2048, "date_captured": "2013-10-04 13:31:53"},
        {"id": "img2", "height": 1494, "width": 2048, "date_captured": "2013-10-04 02:10:00"},
        {"id": "img3", "height": 1080, "width": 1920, "date_captured": "2013-10-04 09:00:00"},
    ])

    result = metadata_survey(images)

    assert result["resolution_counts"][(1494, 2048)] == 2
    assert result["resolution_counts"][(1080, 1920)] == 1


def test_metadata_survey_day_night_proxy_from_hour():
    images = pd.DataFrame([
        {"id": "img1", "height": 1494, "width": 2048, "date_captured": "2013-10-04 13:31:53"},
        {"id": "img2", "height": 1494, "width": 2048, "date_captured": "2013-10-04 02:10:00"},
    ])

    result = metadata_survey(images)

    assert result["day_count"] == 1
    assert result["night_count"] == 1


def test_metadata_survey_reports_unparseable_dates_separately():
    images = pd.DataFrame([
        {"id": "img1", "height": 1494, "width": 2048, "date_captured": "2013-10-04 13:31:53"},
        {"id": "img2", "height": 1494, "width": 2048, "date_captured": "11 11"},
    ])

    result = metadata_survey(images)

    assert result["unparseable_date_count"] == 1
    assert result["day_count"] + result["night_count"] == 1


def test_aspect_ratio_survey_buckets_landscape_portrait_square():
    images = pd.DataFrame([
        {"id": "img1", "height": 1000, "width": 2000},  # landscape
        {"id": "img2", "height": 2000, "width": 1000},  # portrait
        {"id": "img3", "height": 1000, "width": 1000},  # square
        {"id": "img4", "height": 1494, "width": 2048},  # landscape (real CCT resolution)
    ])

    result = aspect_ratio_survey(images)

    assert result["landscape"] == 2
    assert result["portrait"] == 1
    assert result["square"] == 1


def test_bbox_area_ratio_computes_fraction_of_image_area():
    bboxes = pd.DataFrame([
        {"image_id": "img1", "bbox": [0.0, 0.0, 10.0, 10.0]},  # 100 px^2
        {"image_id": "img2", "bbox": [0.0, 0.0, 50.0, 50.0]},  # 2500 px^2
    ])
    images = pd.DataFrame([
        {"id": "img1", "height": 100, "width": 100},   # 10000 px^2 -> ratio 0.01
        {"id": "img2", "height": 100, "width": 100},   # 10000 px^2 -> ratio 0.25
    ])

    result = bbox_area_ratio(bboxes, images)

    assert result.loc[result["image_id"] == "img1", "area_ratio"].iloc[0] == pytest.approx(0.01)
    assert result.loc[result["image_id"] == "img2", "area_ratio"].iloc[0] == pytest.approx(0.25)
