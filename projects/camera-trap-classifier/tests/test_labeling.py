import pandas as pd
import pytest

from src.classifier.labeling import build_crop_dataframe, match_crop_to_ground_truth


def test_match_crop_to_ground_truth_picks_highest_iou():
    crop_bbox = (0, 0, 10, 10)
    ground_truth = [((0, 0, 10, 10), "fox"), ((5, 5, 10, 10), "coyote")]

    result = match_crop_to_ground_truth(crop_bbox, ground_truth)

    assert result == "fox"


def test_match_crop_to_ground_truth_returns_none_when_no_overlap():
    crop_bbox = (0, 0, 10, 10)
    ground_truth = [((500, 500, 10, 10), "fox")]

    result = match_crop_to_ground_truth(crop_bbox, ground_truth)

    assert result is None


def test_build_crop_dataframe_uses_ground_truth_when_available():
    detections = [{
        "source_image": "a.jpg",
        "crops": [{"crop_file": "a_crop0.jpg", "bbox_detected": [0, 0, 10, 10], "confidence": 0.9}],
    }]
    image_species = {"a.jpg": {"fox", "coyote"}}
    image_ground_truth = {"a.jpg": [((0, 0, 10, 10), "fox"), ((100, 100, 10, 10), "coyote")]}

    result = build_crop_dataframe(detections, image_species, image_ground_truth)

    assert len(result) == 1
    assert result.iloc[0]["species"] == "fox"


def test_build_crop_dataframe_uses_single_species_fallback_without_ground_truth():
    detections = [{
        "source_image": "b.jpg",
        "crops": [{"crop_file": "b_crop0.jpg", "bbox_detected": [0, 0, 10, 10], "confidence": 0.9}],
    }]
    image_species = {"b.jpg": {"deer"}}
    image_ground_truth = {}

    result = build_crop_dataframe(detections, image_species, image_ground_truth)

    assert len(result) == 1
    assert result.iloc[0]["species"] == "deer"


def test_build_crop_dataframe_drops_ambiguous_multi_species_without_ground_truth():
    detections = [{
        "source_image": "c.jpg",
        "crops": [
            {"crop_file": "c_crop0.jpg", "bbox_detected": [0, 0, 10, 10], "confidence": 0.9},
            {"crop_file": "c_crop1.jpg", "bbox_detected": [50, 50, 10, 10], "confidence": 0.8},
        ],
    }]
    image_species = {"c.jpg": {"bobcat", "rabbit"}}
    image_ground_truth = {}

    result = build_crop_dataframe(detections, image_species, image_ground_truth)

    assert len(result) == 0


def test_build_crop_dataframe_drops_crop_with_no_ground_truth_overlap():
    detections = [{
        "source_image": "d.jpg",
        "crops": [{"crop_file": "d_crop0.jpg", "bbox_detected": [900, 900, 10, 10], "confidence": 0.9}],
    }]
    image_species = {"d.jpg": {"squirrel", "bird"}}
    image_ground_truth = {"d.jpg": [((0, 0, 10, 10), "squirrel")]}

    result = build_crop_dataframe(detections, image_species, image_ground_truth)

    assert len(result) == 0
