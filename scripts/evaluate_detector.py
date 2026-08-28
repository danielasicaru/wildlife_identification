"""Computes detector average precision and characterizes missed detections by condition (bbox
size, day/night). Writes reports/detector_evaluation.md."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.characterize import bbox_area_ratio
from src.evaluation.detector_metrics import average_precision
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

# Ground truth is scoped to the processed sample (detection_results), not the full ~63,025-image
# bbox dataset -- otherwise the AP denominator counts ground-truth boxes for images the detector
# never even saw, which would make AP look catastrophically low for no real reason.
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

# --- Missed-detection analysis by bbox size ---
ratios = bbox_area_ratio(bbox_annotations, bbox_images)
ratios["file_name"] = ratios["image_id"].map(image_id_to_file)
# Scoped to the processed sample for the same reason as ground_truth above -- otherwise every
# ground-truth box outside the 633-image sample counts as "missed" for no real reason.
ratios = ratios[ratios["file_name"].isin(processed_files)]
detected_files = {d["image_id"] for d in detections}

ratios["size_bucket"] = pd.cut(
    ratios["area_ratio"], bins=[0, 0.02, 0.1, 1.0], labels=["small (<2%)", "medium (2-10%)", "large (>10%)"]
)
ratios["detected"] = ratios["file_name"].isin(detected_files)
size_recall = ratios.groupby("size_bucket", observed=True)["detected"].agg(["mean", "count"])

# --- Missed-detection analysis by day/night ---
sample_files_with_gt = [f for f in ground_truth if (IMAGES_DIR / f).exists()]
day_night_rows = [
    {
        "file_name": file_name,
        "day_night": day_night_label(IMAGES_DIR / file_name),
        "detected": file_name in detected_files,
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
    size_recall.round(3).to_markdown(),
    "",
]
if day_night_recall is not None:
    lines += [
        "## Missed-detection analysis by day/night (pixel-based)",
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
