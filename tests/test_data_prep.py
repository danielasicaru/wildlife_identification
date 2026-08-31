import json

from src.classifier.data_prep import build_labeled_crop_df


def _write_fixtures(tmp_path):
    detections = [{
        "source_image": "a.jpg",
        "crops": [{"crop_file": "a_crop0.jpg", "bbox_detected": [0, 0, 10, 10], "confidence": 0.9}],
    }, {
        "source_image": "b.jpg",
        "crops": [{"crop_file": "b_crop0.jpg", "bbox_detected": [0, 0, 10, 10], "confidence": 0.9}],
    }]
    annotations_data = {
        "images": [
            {"id": "img_a", "file_name": "a.jpg", "location": "1", "date_captured": "2013-10-04 13:31:53"},
            {"id": "img_b", "file_name": "b.jpg", "location": "2", "date_captured": "2013-10-04 13:31:53"},
        ],
        "annotations": [
            {"id": "ann_a", "image_id": "img_a", "category_id": 1},
            {"id": "ann_b", "image_id": "img_b", "category_id": 1},
        ],
        "categories": [{"id": 1, "name": "fox"}],
        "info": {"version": "test"},
    }
    bbox_data = {"images": [], "annotations": []}

    detections_path = tmp_path / "detections.json"
    annotations_path = tmp_path / "annotations.json"
    bbox_path = tmp_path / "bboxes.json"
    detections_path.write_text(json.dumps(detections))
    annotations_path.write_text(json.dumps(annotations_data))
    bbox_path.write_text(json.dumps(bbox_data))
    return detections_path, annotations_path, bbox_path


def test_build_labeled_crop_df_returns_crop_df_and_images_df(tmp_path):
    detections_path, annotations_path, bbox_path = _write_fixtures(tmp_path)

    crop_df, images_df = build_labeled_crop_df(detections_path, annotations_path, bbox_path, min_samples_per_species=1)

    assert list(crop_df["species"]) == ["fox", "fox"]
    assert set(images_df["file_name"]) == {"a.jpg", "b.jpg"}


def test_build_labeled_crop_df_drops_species_below_min_samples(tmp_path):
    detections_path, annotations_path, bbox_path = _write_fixtures(tmp_path)

    crop_df, _ = build_labeled_crop_df(detections_path, annotations_path, bbox_path, min_samples_per_species=3)

    assert len(crop_df) == 0
