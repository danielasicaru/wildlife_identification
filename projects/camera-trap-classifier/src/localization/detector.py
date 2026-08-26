"""MegaDetector integration: model loading, inference, and pure filtering/geometry helpers."""


def bbox_to_absolute(bbox_normalized: list[float], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """Convert a MegaDetector [x, y, w, h] normalized bbox to absolute pixel [x, y, w, h].

    MegaDetector reports bboxes normalized to [0, 1]; ground-truth CCT annotations use absolute
    pixel coordinates. Both formats are needed to compare detections against ground truth.
    """
    x, y, w, h = bbox_normalized
    return (
        round(x * image_width),
        round(y * image_height),
        round(w * image_width),
        round(h * image_height),
    )


def filter_animal_detections(detections: list[dict], min_confidence: float = 0.2) -> list[dict]:
    """Keep only category "1" (animal) detections at or above min_confidence.

    MegaDetector also reports "person" (2) and "vehicle" (3) detections, which aren't relevant to
    species classification and would otherwise get cropped and fed to the classifier as if they
    were animals. 0.2 is MegaDetector's own documented "typical" confidence threshold for v5a.
    """
    return [d for d in detections if d["category"] == "1" and d["conf"] >= min_confidence]
