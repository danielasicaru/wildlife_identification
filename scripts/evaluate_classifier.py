"""Evaluates the best backbone from the most recent training run on the held-out test split:
per-class precision/recall/F1, confusion matrix, day/night and per-site error segmentation, and a
qualitative failure list. Writes reports/classifier_evaluation.md and
reports/confusion_matrix.png."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import matplotlib.pyplot as plt
import mlflow
import torch

from src.classifier.dataset import CropDataset
from src.classifier.labeling import build_crop_dataframe
from src.classifier.models import BACKBONES, build_model
from src.classifier.split import group_images_by_near_duplicates, split_groups
from src.data.loader import load_annotations, merge_categories
from src.evaluation.classifier_metrics import confusion_matrix_df, per_class_report
from src.evaluation.segmentation import build_site_lookup, day_night_label
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS_PATH = ROOT / "data" / "localization" / "detections.json"
CROPS_DIR = ROOT / "data" / "localization" / "crops"
ANNOTATIONS_PATH = ROOT / "data" / "raw" / "caltech_images_20210113.json"
BBOX_PATH = ROOT / "data" / "raw" / "caltech_bboxes_20200316.json"
IMAGES_DIR = ROOT / "data" / "raw" / "images"
NEAR_DUPLICATES_PATH = ROOT / "data" / "near_duplicates.json"
REPORT_PATH = ROOT / "reports" / "classifier_evaluation.md"
CONFUSION_MATRIX_PATH = ROOT / "reports" / "confusion_matrix.png"
NON_SPECIES = {"empty", "car"}

train_config = load_config(ROOT / "configs" / "train_classifier.yaml")
SEED = train_config["seed"]
MIN_SAMPLES_PER_SPECIES = train_config["min_samples_per_species"]

# --- Rebuild the exact same labeled/split crop_df train_classifier.py used ---
with open(DETECTIONS_PATH, encoding="utf-8") as f:
    detections = json.load(f)
images_df, annotations_df, categories_df = load_annotations(ANNOTATIONS_PATH)
merged = merge_categories(annotations_df, categories_df)
merged = merged.merge(images_df[["id", "file_name"]], left_on="image_id", right_on="id", suffixes=("", "_img"))
merged = merged[~merged["name"].isin(NON_SPECIES)]
image_species = merged.groupby("file_name")["name"].apply(set).to_dict()

with open(BBOX_PATH, encoding="utf-8") as f:
    bbox_data = json.load(f)
bbox_images = {im["id"]: im["file_name"] for im in bbox_data["images"]}
image_ground_truth: dict[str, list] = {}
for ann in bbox_data["annotations"]:
    file_name = bbox_images.get(ann["image_id"])
    species = image_species.get(file_name)
    if file_name is None or not species or len(species) != 1:
        continue
    bbox_abs = tuple(round(v) for v in ann["bbox"])
    image_ground_truth.setdefault(file_name, []).append((bbox_abs, next(iter(species))))

crop_df = build_crop_dataframe(detections, image_species, image_ground_truth)
species_counts = crop_df["species"].value_counts()
crop_df = crop_df[crop_df["species"].map(species_counts) >= MIN_SAMPLES_PER_SPECIES].reset_index(drop=True)

if not NEAR_DUPLICATES_PATH.exists():
    # No fallback recompute here (unlike train_classifier.py) -- this script must reconstruct the
    # exact split a checkpoint was trained on, and a fresh recomputation could diverge from it.
    raise SystemExit(
        f"{NEAR_DUPLICATES_PATH} not found -- required to reconstruct the exact split the "
        "checkpoint was trained on. Run scripts/generate_quality_report.py first."
    )
with open(NEAR_DUPLICATES_PATH, encoding="utf-8") as f:
    duplicate_pairs = [tuple(pair) for pair in json.load(f)]
groups = group_images_by_near_duplicates(crop_df["source_image"].unique().tolist(), duplicate_pairs)
crop_df["group_id"] = crop_df["source_image"].map(groups)
crop_df["split"] = split_groups(crop_df, seed=SEED)

train_species = set(crop_df[crop_df["split"] == "train"]["species"].unique())
crop_df = crop_df[crop_df["species"].isin(train_species)].reset_index(drop=True)
test_df = crop_df[crop_df["split"] == "test"].reset_index(drop=True)

if test_df.empty:
    raise SystemExit("Test split is empty -- nothing to evaluate.")

# --- Load the best backbone from the most recent training run ---
mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
runs = mlflow.search_runs(experiment_names=["camera-trap-classifier"], order_by=["start_time DESC"])
if runs.empty:
    raise SystemExit("No MLflow runs found -- run scripts/train_classifier.py first.")

# Only compare within the most recent batch of runs (one per backbone) -- older historical runs
# used different code/data and aren't a fair or even loadable comparison (their checkpoints may
# not exist if they predate checkpoint saving).
latest_batch = runs.head(len(BACKBONES))
best_run = latest_batch.sort_values("metrics.final_val_accuracy", ascending=False).iloc[0]
backbone = best_run["tags.mlflow.runName"]
print(f"Evaluating best run from the latest batch: {backbone} (val_accuracy={best_run['metrics.final_val_accuracy']:.3f})")

client = mlflow.tracking.MlflowClient()
artifact_dir = Path(client.download_artifacts(best_run["run_id"], ""))
species_to_index = json.loads((artifact_dir / "species_to_index.json").read_text(encoding="utf-8"))
index_to_species = {v: k for k, v in species_to_index.items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(backbone, num_classes=len(species_to_index), pretrained=False).to(device)
model.load_state_dict(torch.load(artifact_dir / f"{backbone}.pt", map_location=device))
model.eval()

# --- Run on the test split ---
test_dataset = CropDataset(test_df, CROPS_DIR, species_to_index, is_train=False)
y_true, y_pred, confidences = [], [], []
misclassified = []

with torch.no_grad():
    for i in range(len(test_dataset)):
        image, label_index = test_dataset[i]
        row = test_df.iloc[i]
        logits = model(image.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)
        pred_index = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, pred_index].item())

        true_species = index_to_species[label_index]
        pred_species = index_to_species[pred_index]
        y_true.append(true_species)
        y_pred.append(pred_species)
        confidences.append(confidence)

        if true_species != pred_species:
            misclassified.append({
                "crop_file": row["crop_file"], "true": true_species, "predicted": pred_species,
                "confidence": confidence,
            })

labels = sorted(species_to_index.keys())
report = per_class_report(y_true, y_pred, labels)
matrix = confusion_matrix_df(y_true, y_pred, labels)
overall_accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

# --- Day/night and per-site segmentation ---
site_lookup = build_site_lookup(images_df)
test_df = test_df.copy()
test_df["correct"] = [t == p for t, p in zip(y_true, y_pred)]
test_df["day_night"] = [day_night_label(IMAGES_DIR / row["source_image"]) for _, row in test_df.iterrows()]
test_df["site"] = test_df["source_image"].map(site_lookup)

day_night_accuracy = test_df.groupby("day_night")["correct"].agg(["mean", "count"])
site_errors = test_df[~test_df["correct"]]["site"].value_counts()

# --- Confusion matrix plot ---
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(matrix.values, cmap="Blues")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion matrix -- {backbone} (test accuracy {overall_accuracy:.1%})")
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH)
plt.close(fig)

# --- Report ---
lines = [
    "# Classifier Evaluation",
    "",
    f"Best run (latest batch): **{backbone}** (val_accuracy={best_run['metrics.final_val_accuracy']:.3f})",
    f"Test set: {len(test_df)} crops across {len(labels)} species",
    f"Overall test accuracy: **{overall_accuracy:.1%}**",
    "",
    "## Per-class metrics",
    "",
    report.round(3).to_markdown(),
    "",
    "## Confusion matrix",
    "",
    "See `reports/confusion_matrix.png`.",
    "",
    "## Day/night segmentation",
    "",
    day_night_accuracy.round(3).to_markdown(),
    "",
    "## Per-site errors (raw counts, sample too fragmented for accuracy-rate claims)",
    "",
    f"{len(site_errors)} sites had at least one misclassification, out of "
    f"{test_df['site'].nunique()} sites represented in the {len(test_df)}-crop test set "
    f"(most sites have 1-7 test examples -- too few for a reliable per-site accuracy rate).",
    "",
    "## Failure analysis (qualitative)",
    "",
]
for item in misclassified:
    lines.append(f"- `{item['crop_file']}`: true={item['true']}, predicted={item['predicted']} (confidence={item['confidence']:.2f})")
lines += [
    "",
    "Occlusion segmentation is not included: only 20 images have manual occlusion tags (from the "
    "dataset characterization stage), against 88 test crops from different source images -- "
    "expected overlap is near zero, so a segmentation on that basis wouldn't be meaningful at "
    "this sample size.",
]

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Overall test accuracy: {overall_accuracy:.1%}")
print(f"Report: {REPORT_PATH}")
print(f"Confusion matrix: {CONFUSION_MATRIX_PATH}")
