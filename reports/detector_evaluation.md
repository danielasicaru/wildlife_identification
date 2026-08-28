# Detector Evaluation

**Average Precision (IoU >= 0.5): 0.535**

617 detections across 633 images compared against 358 ground-truth animal boxes in 314 annotated images.

## Missed-detection analysis by animal size (fraction of frame)

Per-box, IoU-matched (IoU >= 0.5) -- whether this specific ground-truth box was detected, not just whether the image got any detection at all.

| size_bucket    |   mean |   count |
|:---------------|-------:|--------:|
| small (<2%)    |  0.882 |     153 |
| medium (2-10%) |  1     |     156 |
| large (>10%)   |  0.816 |      49 |

## Missed-detection analysis by day/night (pixel-based)

Image-level: whether the image got at least one detection at all, not IoU-matched per box (day/night is inherently an image-level property, unlike animal size below).

| day_night   |   mean |   count |
|:------------|-------:|--------:|
| day         |  0.893 |     131 |
| night       |  0.995 |     183 |

This extends the localization stage's recall sanity check (see reports/localization.md) with a proper precision-recall-integrated AP metric and a breakdown by the conditions identified during dataset characterization as likely failure modes (small/distant animals, night IR captures).