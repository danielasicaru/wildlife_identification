"""Trains and compares ResNet50, EfficientNet-B0, and ViT-B/16 on MegaDetector crops, with
class-weighted loss and MLflow tracking (local file store)."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import numpy as np
import pandas as pd
import torch

from src.classifier.data_prep import build_labeled_crop_df
from src.classifier.prediction import predict_test_set
from src.classifier.split import group_images_by_near_duplicates, split_groups
from src.classifier.training_run import train_and_compare_backbones
from src.data.quality import find_near_duplicates
from src.evaluation.classifier_metrics import per_class_report
from src.utils.config import load_config
from src.utils.mlflow_tracking import mlflow_tracking_uri

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS_PATH = ROOT / "data" / "localization" / "detections.json"
CROPS_DIR = ROOT / "data" / "localization" / "crops"
ANNOTATIONS_PATH = ROOT / "data" / "raw" / "caltech_images_20210113.json"
BBOX_PATH = ROOT / "data" / "raw" / "caltech_bboxes_20200316.json"
IMAGES_DIR = ROOT / "data" / "raw" / "images"
NEAR_DUPLICATES_PATH = ROOT / "data" / "near_duplicates.json"
DATA_MANIFEST_PATH = ROOT / "reports" / "data_manifest.json"
CONFIG_PATH = ROOT / "configs" / "train_classifier.yaml"
CHECKPOINT_DIR = ROOT / "data" / "checkpoints"

config = load_config(CONFIG_PATH)
SEED = config["seed"]
EPOCHS = config["epochs"]
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = config["learning_rate"]
MIN_SAMPLES_PER_SPECIES = config["min_samples_per_species"]
EARLY_STOPPING_PATIENCE = config["early_stopping_patience"]
BACKBONES = tuple(config["backbones"])

if EPOCHS < 1:
    raise SystemExit(f"configs/train_classifier.yaml: epochs must be >= 1, got {EPOCHS}")

for path in (DETECTIONS_PATH, ANNOTATIONS_PATH, BBOX_PATH):
    if not path.exists():
        raise SystemExit(f"{path} not found -- run the localization and download scripts first.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# Narrows (doesn't fully eliminate) GPU nondeterminism from cuDNN's algorithm selection; full
# determinism would need torch.use_deterministic_algorithms(True), which can raise on ops with
# no deterministic implementation.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Per-crop labels ---
crop_df, _ = build_labeled_crop_df(DETECTIONS_PATH, ANNOTATIONS_PATH, BBOX_PATH, MIN_SAMPLES_PER_SPECIES)
print(f"{len(crop_df)} labeled crops across {crop_df['species'].nunique()} species")

# --- Near-duplicate-aware split ---
if NEAR_DUPLICATES_PATH.exists():
    # Reuse the pairs already computed by generate_quality_report.py rather than re-running the
    # O(n^2) perceptual-hash comparison over the same sample images.
    with open(NEAR_DUPLICATES_PATH, encoding="utf-8") as f:
        duplicate_pairs = [tuple(pair) for pair in json.load(f)]
else:
    sample_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    duplicate_pairs = [(a.name, b.name) for a, b in find_near_duplicates(sample_paths)]
groups = group_images_by_near_duplicates(crop_df["source_image"].unique().tolist(), duplicate_pairs)
crop_df["group_id"] = crop_df["source_image"].map(groups)
crop_df["split"] = split_groups(crop_df, seed=SEED)

train_df = crop_df[crop_df["split"] == "train"].reset_index(drop=True)

# The split is group-level, not per-class stratified, so a species can end up entirely outside
# train_df -- drop it everywhere rather than scoring a class the model never saw.
train_species = set(train_df["species"].unique())
species_without_train_examples = set(crop_df["species"].unique()) - train_species
if species_without_train_examples:
    print(f"Dropping {len(species_without_train_examples)} species with zero train examples: {sorted(species_without_train_examples)}")
    crop_df = crop_df[crop_df["species"].isin(train_species)].reset_index(drop=True)
    train_df = crop_df[crop_df["split"] == "train"].reset_index(drop=True)

val_df = crop_df[crop_df["split"] == "val"].reset_index(drop=True)
test_count = int((crop_df["split"] == "test").sum())
print(f"train={len(train_df)}, val={len(val_df)}, test={test_count}")

if train_df.empty or val_df.empty:
    raise SystemExit("Train or val split is empty -- not enough labeled crops to proceed.")

species_to_index = {s: i for i, s in enumerate(sorted(crop_df["species"].unique()))}
index_to_species = {i: s for s, i in species_to_index.items()}
labels = sorted(species_to_index.keys())

mlflow.set_tracking_uri(mlflow_tracking_uri(ROOT))
mlflow.set_experiment("camera-trap-classifier")

# Per-class validation accuracy at every epoch (not just a final-epoch snapshot), so
# reports/confusion_over_epochs.md can show whether specific classes' errors are stable,
# improving, or degrading over the course of training.
epoch_class_accuracy: dict[str, list[dict]] = {backbone: [] for backbone in BACKBONES}


def track_per_class_accuracy(backbone: str, epoch: int, model) -> None:
    model.eval()
    predictions = predict_test_set(model, val_df, CROPS_DIR, species_to_index, index_to_species, device)
    recall_by_class = per_class_report(predictions["true"].tolist(), predictions["predicted"].tolist(), labels)["recall"]
    epoch_class_accuracy[backbone].append({"epoch": epoch, **recall_by_class.to_dict()})


artifact_paths = [CONFIG_PATH] + ([DATA_MANIFEST_PATH] if DATA_MANIFEST_PATH.exists() else [])
results = train_and_compare_backbones(
    train_df, val_df, CROPS_DIR, species_to_index, BACKBONES, SEED, EPOCHS, BATCH_SIZE,
    LEARNING_RATE, EARLY_STOPPING_PATIENCE, device, CHECKPOINT_DIR,
    mlflow_params={}, artifact_paths=artifact_paths, on_epoch_end=track_per_class_accuracy,
)

print("\nComparison (final val accuracy):")
for backbone, metrics in results.items():
    print(f"  {backbone}: {metrics['accuracy']:.3f}")

# --- Per-class accuracy over epochs report ---
REPORT_DIR = ROOT / "reports"
for backbone, rows in epoch_class_accuracy.items():
    pd.DataFrame(rows).set_index("epoch").to_csv(REPORT_DIR / f"confusion_over_epochs_{backbone}.csv")

best_backbone = max(results, key=lambda b: results[b]["accuracy"])
best_epochs_df = pd.DataFrame(epoch_class_accuracy[best_backbone]).set_index("epoch")
midpoint = len(best_epochs_df) // 2
first_half_mean = best_epochs_df.iloc[:midpoint].mean()
second_half_mean = best_epochs_df.iloc[midpoint:].mean()
accuracy_change = (second_half_mean - first_half_mean).sort_values().round(3)

epochs_report_lines = [
    "# Per-Class Accuracy Over Training Epochs",
    "",
    f"Validation-set per-class accuracy (recall) tracked at every epoch during training for "
    f"**{best_backbone}** (this run's best backbone), not just a final-epoch snapshot -- shows "
    "which classes' errors are stable, improving, or degrading as training progresses. The "
    "validation set is small (90 crops across 19 species, many classes with single-digit "
    "support), so per-class accuracy swings by 50 percentage points on a single flipped "
    "prediction -- read trends here as directional, not precise.",
    "",
    f"{len(best_epochs_df)} epochs tracked. Accuracy change from the first half of training to "
    "the second half (second-half mean minus first-half mean; negative = got worse as training "
    "progressed):",
    "",
    accuracy_change.to_frame("accuracy_change").to_markdown(),
    "",
    "Full per-epoch, per-class accuracy for every backbone: "
    "`reports/confusion_over_epochs_<backbone>.csv`.",
]
with open(REPORT_DIR / "confusion_over_epochs.md", "w", encoding="utf-8") as f:
    f.write("\n".join(epochs_report_lines))
print(f"Report: {REPORT_DIR / 'confusion_over_epochs.md'}")
