# Detector Evaluation

**Average Precision (IoU >= 0.5): 0.535**
**mAP@[0.5:0.95] (COCO-style, averaged over IoU 0.50-0.95 in steps of 0.05): 0.381**

617 detections across 633 images compared against 358 ground-truth animal boxes in 314 annotated images.

## AP per IoU threshold

mAP@[0.5:0.95] averages AP across these ten thresholds instead of reporting a single value at IoU >= 0.5 -- a box that loosely overlaps an animal counts the same as a tightly-fitted one under the single-threshold metric above; this shows how much AP drops as the overlap requirement gets stricter.

|   iou_threshold |    ap |
|----------------:|------:|
|            0.5  | 0.535 |
|            0.55 | 0.53  |
|            0.6  | 0.523 |
|            0.65 | 0.503 |
|            0.7  | 0.488 |
|            0.75 | 0.438 |
|            0.8  | 0.373 |
|            0.85 | 0.259 |
|            0.9  | 0.127 |
|            0.95 | 0.034 |

## Missed-detection analysis by animal size (fraction of frame)

Per-box, IoU-matched (IoU >= 0.5) -- whether this specific ground-truth box was detected, not just whether the image got any detection at all.

| size_bucket    |   mean |   count |
|:---------------|-------:|--------:|
| small (<2%)    |  0.882 |     153 |
| medium (2-10%) |  1     |     156 |
| large (>10%)   |  0.816 |      49 |

## Missed-detection analysis by day/night (pixel-based)

Aggregated per image (day/night is inherently an image-level property, unlike animal size below), but "detected" still means at least one of that image's ground-truth boxes was IoU-matched -- not just that the image has some detection.

| day_night   |   mean |   count |
|:------------|-------:|--------:|
| day         |  0.87  |     131 |
| night       |  0.973 |     183 |

This extends the localization stage's recall sanity check (see reports/localization.md) with a proper precision-recall-integrated AP metric and a breakdown by the conditions identified during dataset characterization as likely failure modes (small/distant animals, night IR captures).