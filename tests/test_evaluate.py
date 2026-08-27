import pytest

from src.localization.evaluate import iou, match_detections_to_ground_truth


def test_iou_identical_boxes_is_one():
    box = (10, 10, 100, 100)

    assert iou(box, box) == pytest.approx(1.0)


def test_iou_non_overlapping_boxes_is_zero():
    box_a = (0, 0, 10, 10)
    box_b = (100, 100, 10, 10)

    assert iou(box_a, box_b) == 0.0


def test_iou_partial_overlap():
    box_a = (0, 0, 10, 10)
    box_b = (5, 5, 10, 10)

    assert iou(box_a, box_b) == pytest.approx(25 / 175)


def test_match_detections_recall_all_matched():
    detections = [(0, 0, 10, 10), (50, 50, 10, 10)]
    ground_truth = [(0, 0, 10, 10), (50, 50, 10, 10)]

    result = match_detections_to_ground_truth(detections, ground_truth, iou_threshold=0.5)

    assert result["recall"] == pytest.approx(1.0)
    assert result["matched_count"] == 2
    assert result["ground_truth_count"] == 2


def test_match_detections_recall_partial():
    detections = [(0, 0, 10, 10)]
    ground_truth = [(0, 0, 10, 10), (500, 500, 10, 10)]

    result = match_detections_to_ground_truth(detections, ground_truth, iou_threshold=0.5)

    assert result["recall"] == pytest.approx(0.5)
    assert result["matched_count"] == 1
    assert result["ground_truth_count"] == 2


def test_match_detections_recall_zero_ground_truth_is_none():
    result = match_detections_to_ground_truth([], [], iou_threshold=0.5)

    assert result["recall"] is None
    assert result["ground_truth_count"] == 0
