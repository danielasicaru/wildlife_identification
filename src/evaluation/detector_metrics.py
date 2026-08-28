"""Average precision for the single-class (animal) detector, standard continuous-recall AP."""
from src.localization.evaluate import iou


def average_precision(
    detections: list[dict],
    ground_truth: dict[str, list[tuple[int, int, int, int]]],
    iou_threshold: float = 0.5,
) -> float:
    """detections: list of {"image_id", "bbox", "confidence"}, sorted internally by confidence
    descending. ground_truth: image_id -> list of ground-truth boxes. Standard single-class AP:
    precision integrated over the full recall curve as confidence threshold is swept.
    """
    total_ground_truth = sum(len(boxes) for boxes in ground_truth.values())
    if total_ground_truth == 0:
        return 0.0
    if not detections:
        return 0.0

    sorted_detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    matched = {image_id: [False] * len(boxes) for image_id, boxes in ground_truth.items()}

    tp = [0] * len(sorted_detections)
    fp = [0] * len(sorted_detections)

    for i, detection in enumerate(sorted_detections):
        image_id = detection["image_id"]
        gt_boxes = ground_truth.get(image_id, [])

        best_iou = 0.0
        best_j = -1
        for j, gt_box in enumerate(gt_boxes):
            if matched.get(image_id, [])[j]:
                continue
            score = iou(detection["bbox"], gt_box)
            if score > best_iou:
                best_iou = score
                best_j = j

        if best_iou >= iou_threshold and best_j >= 0:
            tp[i] = 1
            matched[image_id][best_j] = True
        else:
            fp[i] = 1

    cum_tp = [sum(tp[: i + 1]) for i in range(len(tp))]
    cum_fp = [sum(fp[: i + 1]) for i in range(len(fp))]
    precisions = [t / (t + f) if (t + f) > 0 else 0.0 for t, f in zip(cum_tp, cum_fp)]
    recalls = [t / total_ground_truth for t in cum_tp]

    # Standard AP: precision envelope (each precision replaced by the max precision at that
    # recall or higher), integrated via trapezoid over recall.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    prev_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        ap += precision * (recall - prev_recall)
        prev_recall = recall

    return ap
