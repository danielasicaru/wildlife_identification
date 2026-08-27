"""Compares MegaDetector detections (from run_localization.py) against the existing CCT
ground-truth bounding boxes, for the subset of sample images that have ground-truth annotations.
Writes reports/localization.md (gitignored -- regenerate locally)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.localization.evaluate import match_detections_to_ground_truth

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS_PATH = ROOT / "data" / "localization" / "detections.json"
BBOX_PATH = ROOT / "data" / "raw" / "caltech_bboxes_20200316.json"
REPORT_PATH = ROOT / "reports" / "localization.md"

if not DETECTIONS_PATH.exists():
    raise SystemExit(f"{DETECTIONS_PATH} not found -- run scripts/run_localization.py first.")
if not BBOX_PATH.exists():
    raise SystemExit(f"{BBOX_PATH} not found -- ground-truth bboxes required for recall comparison.")

with open(DETECTIONS_PATH, encoding="utf-8") as f:
    detection_results = json.load(f)

with open(BBOX_PATH, encoding="utf-8") as f:
    bbox_data = json.load(f)

gt_annotations = pd.DataFrame(bbox_data["annotations"])
gt_images = pd.DataFrame(bbox_data["images"])[["id", "file_name"]]
if gt_images["file_name"].duplicated().any():
    raise SystemExit(
        "Ground-truth images have duplicate file_name values -- the filename-based merge below "
        "would silently pool boxes from unrelated images. Join on image id instead."
    )
gt_by_file = gt_annotations.merge(gt_images, left_on="image_id", right_on="id")

per_image_recall = []
for result in detection_results:
    filename = result["source_image"]
    gt_rows = gt_by_file[gt_by_file["file_name"] == filename]
    if gt_rows.empty:
        continue  # this sample image has no ground-truth bbox annotation

    ground_truth_boxes = [tuple(round(v) for v in bbox) for bbox in gt_rows["bbox"]]
    # Compare against the raw detected box, not the classifier-input expanded crop box --
    # the expansion margin isn't part of what the detector actually found.
    detected_boxes = [tuple(c["bbox_detected"]) for c in result["crops"]]

    match_result = match_detections_to_ground_truth(detected_boxes, ground_truth_boxes, iou_threshold=0.5)
    per_image_recall.append({"file": filename, **match_result})

total_matched = sum(r["matched_count"] for r in per_image_recall)
total_ground_truth = sum(r["ground_truth_count"] for r in per_image_recall)
overall_recall = total_matched / total_ground_truth if total_ground_truth else None

lines = [
    "# Localization report",
    "",
    f"{len(per_image_recall)} sample images had ground-truth bbox annotations to compare against "
    f"(of {len(detection_results)} total processed).",
    "",
]
if overall_recall is not None:
    lines.append(
        f"**Overall recall (IoU >= 0.5): {overall_recall:.1%}** "
        f"({total_matched}/{total_ground_truth} ground-truth animals detected)"
    )
else:
    lines.append("No ground-truth-annotated images in this sample.")

lines += [
    "",
    "This is a sanity check on a small sample, not a full detector evaluation (mAP, "
    "missed-detection analysis by condition) -- that's deferred to the evaluation stage once "
    "training/validation splits exist.",
]

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
