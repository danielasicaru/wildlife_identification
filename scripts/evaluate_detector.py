"""Computes detector average precision and characterizes missed detections by condition (bbox
size, day/night). Writes reports/detector_evaluation.md."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.evaluation.detector_metrics import average_precision, per_box_detected
from src.evaluation.segmentation import day_night_label

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS_PATH = ROOT / "data" / "localization" / "detections.json"
BBOX_PATH = ROOT / "data" / "raw" / "caltech_bboxes_20200316.json"
IMAGES_DIR = ROOT / "data" / "raw" / "images"
REPORT_PATH = ROOT / "reports" / "detector_evaluation.md"

if not DETECTIONS_PATH.exists() or not BBOX_PATH.exists():
    raise SystemExit("Run scripts/run_localization.py first, and ensure ground-truth bboxes are downloaded.")

with open(DETECTIONS_PATH, encoding="utf-8") as f:
    detection_results = json.load(f)
with open(BBOX_PATH, encoding="utf-8") as f:
    bbox_data = json.load(f)

bbox_annotations = pd.DataFrame(bbox_data["annotations"])
bbox_images = pd.DataFrame(bbox_data["images"])[["id", "file_name", "height", "width"]]
image_id_to_file = bbox_images.set_index("id")["file_name"].to_dict()

# Scoped to the processed sample, not the full ~63,025-image bbox dataset -- otherwise the AP
# denominator counts ground truth for images the detector never saw.
processed_files = {result["source_image"] for result in detection_results}

ground_truth: dict[str, list] = {}
for _, row in bbox_annotations.iterrows():
    file_name = image_id_to_file.get(row["image_id"])
    if file_name is None or file_name not in processed_files:
        continue
    ground_truth.setdefault(file_name, []).append(tuple(round(v) for v in row["bbox"]))

detections = [
    {"image_id": result["source_image"], "bbox": tuple(crop["bbox_detected"]), "confidence": crop["confidence"]}
    for result in detection_results
    for crop in result["crops"]
]

ap = average_precision(detections, ground_truth, iou_threshold=0.5)

# Per-box IoU-matched status, reused below by both breakdowns -- a raw "did this image get any
# detection" flag would wrongly credit a false-positive-only image as a real detection.
box_matches = per_box_detected(detections, ground_truth, iou_threshold=0.5)

# --- Missed-detection analysis by bbox size (per-box) ---
image_dims = bbox_images.set_index("file_name")[["height", "width"]].to_dict("index")

size_rows = []
for file_name, boxes in ground_truth.items():
    dims = image_dims[file_name]
    matches = box_matches[file_name]
    for bbox, detected in zip(boxes, matches):
        area_ratio = (bbox[2] * bbox[3]) / (dims["height"] * dims["width"])
        size_rows.append({"file_name": file_name, "area_ratio": area_ratio, "detected": detected})

ratios = pd.DataFrame(size_rows)
ratios["size_bucket"] = pd.cut(
    ratios["area_ratio"], bins=[0, 0.02, 0.1, 1.0], labels=["small (<2%)", "medium (2-10%)", "large (>10%)"]
)
size_recall = ratios.groupby("size_bucket", observed=True)["detected"].agg(["mean", "count"])

# --- Missed-detection analysis by day/night ---
sample_files_with_gt = [f for f in ground_truth if (IMAGES_DIR / f).exists()]
day_night_rows = [
    {
        "file_name": file_name,
        "day_night": day_night_label(IMAGES_DIR / file_name),
        "detected": any(box_matches[file_name]),
    }
    for file_name in sample_files_with_gt
]
day_night_df = pd.DataFrame(day_night_rows)
day_night_recall = day_night_df.groupby("day_night")["detected"].agg(["mean", "count"]) if not day_night_df.empty else None

lines = [
    "# Detector Evaluation",
    "",
    f"**Average Precision (IoU >= 0.5): {ap:.3f}**",
    "",
    f"{len(detections)} detections across {len(detection_results)} images compared against "
    f"{sum(len(v) for v in ground_truth.values())} ground-truth animal boxes in "
    f"{len(ground_truth)} annotated images.",
    "",
    "## Missed-detection analysis by animal size (fraction of frame)",
    "",
    "Per-box, IoU-matched (IoU >= 0.5) -- whether this specific ground-truth box was detected, "
    "not just whether the image got any detection at all.",
    "",
    size_recall.round(3).to_markdown(),
    "",
]
if day_night_recall is not None:
    lines += [
        "## Missed-detection analysis by day/night (pixel-based)",
        "",
        "Aggregated per image (day/night is inherently an image-level property, unlike animal "
        "size below), but \"detected\" still means at least one of that image's ground-truth "
        "boxes was IoU-matched -- not just that the image has some detection.",
        "",
        day_night_recall.round(3).to_markdown(),
        "",
    ]
lines += [
    "This extends the localization stage's recall sanity check (see reports/localization.md) "
    "with a proper precision-recall-integrated AP metric and a breakdown by the conditions "
    "identified during dataset characterization as likely failure modes (small/distant animals, "
    "night IR captures).",
]

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Average Precision (IoU >= 0.5): {ap:.3f}")
print(f"Report: {REPORT_PATH}")
