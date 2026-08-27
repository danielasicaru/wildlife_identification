from pathlib import Path

import pytest
from PIL import Image

from src.localization.detector import bbox_to_absolute, filter_animal_detections, load_detector, run_detection

SAMPLE_IMAGES_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "images"


def test_bbox_to_absolute_converts_normalized_to_pixels():
    result = bbox_to_absolute([0.1, 0.2, 0.3, 0.4], image_width=1000, image_height=500)

    assert result == (100, 100, 300, 200)


def test_filter_animal_detections_excludes_non_animal_categories():
    detections = [
        {"category": "1", "conf": 0.9, "bbox": [0, 0, 0.1, 0.1]},
        {"category": "2", "conf": 0.95, "bbox": [0, 0, 0.1, 0.1]},
        {"category": "3", "conf": 0.99, "bbox": [0, 0, 0.1, 0.1]},
    ]

    result = filter_animal_detections(detections, min_confidence=0.2)

    assert len(result) == 1
    assert result[0]["category"] == "1"


def test_filter_animal_detections_excludes_low_confidence():
    detections = [
        {"category": "1", "conf": 0.1, "bbox": [0, 0, 0.1, 0.1]},
        {"category": "1", "conf": 0.5, "bbox": [0, 0, 0.1, 0.1]},
    ]

    result = filter_animal_detections(detections, min_confidence=0.2)

    assert len(result) == 1
    assert result[0]["conf"] == 0.5


@pytest.mark.skipif(not SAMPLE_IMAGES_DIR.exists(), reason="requires downloaded sample images")
def test_run_detection_returns_detections_for_real_image():
    detector = load_detector()
    sample_path = sorted(SAMPLE_IMAGES_DIR.glob("*.jpg"))[0]
    image = Image.open(sample_path)

    result = run_detection(detector, image, image_id=str(sample_path))

    assert "detections" in result
    assert isinstance(result["detections"], list)
