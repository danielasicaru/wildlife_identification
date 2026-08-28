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


def per_box_detected(
    detections: list[dict],
    ground_truth: dict[str, list[tuple[int, int, int, int]]],
    iou_threshold: float = 0.5,
) -> dict[str, list[bool]]:
    """For each image, whether each of its ground-truth boxes was matched by IoU to at least one
    detection in that image. Returns image_id -> list of booleans, in the same order as
    ground_truth[image_id] -- used for per-box breakdowns (e.g. by animal size) where an
    image-level "did anything get detected" flag would wrongly credit every box in a
    multi-animal image as "detected" even if only one of them actually was.
    """
    detections_by_image: dict[str, list[dict]] = {}
    for detection in detections:
        detections_by_image.setdefault(detection["image_id"], []).append(detection)

    result: dict[str, list[bool]] = {}
    for image_id, gt_boxes in ground_truth.items():
        image_detections = detections_by_image.get(image_id, [])
        result[image_id] = [
            any(iou(detection["bbox"], gt_box) >= iou_threshold for detection in image_detections)
            for gt_box in gt_boxes
        ]

    return result
