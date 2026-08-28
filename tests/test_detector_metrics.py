import pytest

from src.evaluation.detector_metrics import average_precision, per_box_detected


def test_average_precision_perfect_detector_is_one():
    detections = [
        {"image_id": "a", "bbox": (0, 0, 10, 10), "confidence": 0.9},
        {"image_id": "b", "bbox": (0, 0, 10, 10), "confidence": 0.8},
    ]
    ground_truth = {"a": [(0, 0, 10, 10)], "b": [(0, 0, 10, 10)]}

    ap = average_precision(detections, ground_truth, iou_threshold=0.5)

    assert ap == pytest.approx(1.0)


def test_average_precision_no_detections_is_zero():
    detections = []
    ground_truth = {"a": [(0, 0, 10, 10)]}

    ap = average_precision(detections, ground_truth, iou_threshold=0.5)

    assert ap == 0.0


def test_average_precision_penalizes_false_positives():
    detections = [
        {"image_id": "a", "bbox": (0, 0, 10, 10), "confidence": 0.95},  # false positive, no GT here
        {"image_id": "b", "bbox": (0, 0, 10, 10), "confidence": 0.5},   # correct
    ]
    ground_truth = {"a": [], "b": [(0, 0, 10, 10)]}

    ap = average_precision(detections, ground_truth, iou_threshold=0.5)

    assert ap < 1.0


def test_per_box_detected_distinguishes_boxes_within_same_image():
    # Two ground-truth boxes in the same image; only one is actually detected.
    detections = [{"image_id": "a", "bbox": (0, 0, 10, 10), "confidence": 0.9}]
    ground_truth = {"a": [(0, 0, 10, 10), (500, 500, 10, 10)]}

    result = per_box_detected(detections, ground_truth, iou_threshold=0.5)

    assert result["a"] == [True, False]


def test_per_box_detected_false_for_image_with_no_detections():
    detections = []
    ground_truth = {"a": [(0, 0, 10, 10)]}

    result = per_box_detected(detections, ground_truth, iou_threshold=0.5)

    assert result["a"] == [False]
