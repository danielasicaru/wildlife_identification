"""Runs MegaDetector over the downloaded sample images, writing detections and crops to
data/localization/ (gitignored -- regenerate locally with this script)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.localization.crop import crop_to_bbox, expand_bbox
from src.localization.detector import bbox_to_absolute, filter_animal_detections, load_detector, run_detection
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "data" / "raw" / "images"
OUTPUT_DIR = ROOT / "data" / "localization"
CROPS_DIR = OUTPUT_DIR / "crops"
DETECTIONS_PATH = OUTPUT_DIR / "detections.json"

config = load_config(ROOT / "configs" / "run_localization.yaml")
MODEL_NAME = config["model_name"]
MIN_CONFIDENCE = config["min_confidence"]
BOX_EXPANSION_FRACTION = config["box_expansion_fraction"]

if not IMAGES_DIR.exists():
    raise SystemExit(f"No sample images found at {IMAGES_DIR} -- run scripts/download_sample_images.py first.")

CROPS_DIR.mkdir(parents=True, exist_ok=True)

detector = load_detector(MODEL_NAME)
image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
all_results = []
crop_count = 0

for image_path in image_paths:
    image = Image.open(image_path)
    raw_result = run_detection(detector, image, image_id=str(image_path))
    animal_detections = filter_animal_detections(raw_result["detections"], min_confidence=MIN_CONFIDENCE)

    crops = []
    for i, detection in enumerate(animal_detections):
        bbox_abs = bbox_to_absolute(detection["bbox"], image.width, image.height)
        bbox_expanded = expand_bbox(bbox_abs, BOX_EXPANSION_FRACTION, image.width, image.height)
        crop = crop_to_bbox(image, bbox_expanded)

        crop_filename = f"{image_path.stem}_crop{i}.jpg"
        crop.convert("RGB").save(CROPS_DIR / crop_filename)
        crops.append({
            "crop_file": crop_filename,
            "bbox_absolute": list(bbox_expanded),
            # Raw MegaDetector box (pre-expansion) -- kept separately so recall/IoU evaluation
            # compares against what the detector actually found, not the classifier-input margin.
            "bbox_detected": list(bbox_abs),
            "confidence": detection["conf"],
        })
        crop_count += 1

    all_results.append({"source_image": image_path.name, "crops": crops})

with open(DETECTIONS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

images_with_animal = sum(1 for r in all_results if r["crops"])
print(f"{len(image_paths)} images processed, {images_with_animal} with at least one animal detection, {crop_count} crops written")
print(f"Detections: {DETECTIONS_PATH}")
print(f"Crops: {CROPS_DIR}")
