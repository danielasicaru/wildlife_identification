"""IoU-based matching between MegaDetector detections and ground-truth bounding boxes."""


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two absolute-pixel [x, y, w, h] boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    union = aw * ah + bw * bh - intersection
    if union == 0:
        return 0.0
    return intersection / union


def match_detections_to_ground_truth(
    detections: list[tuple[int, int, int, int]],
    ground_truth: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
) -> dict:
    """Greedy one-to-one matching: each ground-truth box counts as matched if some detection
    overlaps it at IoU >= iou_threshold. Returns recall = matched / total ground truth.

    Recall (not precision or mAP) is the primary metric here because a missed detection is an
    end-to-end failure mode (the classifier never sees that animal at all) -- see design spec
    section 5's detector evaluation rationale. Full mAP is deferred to the dedicated evaluation
    stage once a real held-out annotated set is used, not the small localization sanity check here.
    """
    if not ground_truth:
        return {"recall": None, "matched_count": 0, "ground_truth_count": 0}

    matched_count = 0
    for gt_box in ground_truth:
        if any(iou(det_box, gt_box) >= iou_threshold for det_box in detections):
            matched_count += 1

    return {
        "recall": matched_count / len(ground_truth),
        "matched_count": matched_count,
        "ground_truth_count": len(ground_truth),
    }
