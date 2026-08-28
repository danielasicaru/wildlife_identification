"""Trains and compares ResNet50, EfficientNet-B0, and ViT-B/16 on MegaDetector crops, with
class-weighted loss and MLflow tracking (local file store)."""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.classifier.dataset import CropDataset
from src.classifier.engine import EarlyStopping, compute_class_weights, evaluate, train_one_epoch
from src.classifier.labeling import build_crop_dataframe
from src.classifier.models import build_model
from src.classifier.split import group_images_by_near_duplicates, split_groups
from src.data.augmentation import build_sample_weights, minority_species
from src.data.loader import load_annotations, merge_categories
from src.data.quality import find_near_duplicates
from src.utils.config import load_config

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
NON_SPECIES = {"empty", "car"}

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
        continue  # only usable when we independently know the (single) species for this image
    bbox_abs = tuple(round(v) for v in ann["bbox"])
    image_ground_truth.setdefault(file_name, []).append((bbox_abs, next(iter(species))))

crop_df = build_crop_dataframe(detections, image_species, image_ground_truth)
species_counts = crop_df["species"].value_counts()
crop_df = crop_df[crop_df["species"].map(species_counts) >= MIN_SAMPLES_PER_SPECIES].reset_index(drop=True)
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
train_minority = minority_species(train_df["species"].value_counts())

# --- Datasets/loaders ---
train_dataset = CropDataset(train_df, CROPS_DIR, species_to_index, is_train=True, minority_species=train_minority)
val_dataset = CropDataset(val_df, CROPS_DIR, species_to_index, is_train=False)

train_weights = build_sample_weights(train_df["species"], train_df["species"].value_counts())
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- Train + compare backbones ---
mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
mlflow.set_experiment("camera-trap-classifier")

class_weights = compute_class_weights(train_df["species"], species_to_index).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

results = {}
for backbone in BACKBONES:
    # Reseeded to SEED per backbone so each one sees an identical batch order -- backbone is the
    # only thing that varies across iterations.
    sampler = WeightedRandomSampler(
        train_weights, num_samples=len(train_weights), replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

    with mlflow.start_run(run_name=backbone):
        mlflow.log_params({
            "backbone": backbone, "seed": SEED, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "train_size": len(train_df), "val_size": len(val_df), "num_classes": len(species_to_index),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        })
        mlflow.log_artifact(str(CONFIG_PATH))
        if DATA_MANIFEST_PATH.exists():
            mlflow.log_artifact(str(DATA_MANIFEST_PATH))

        model = build_model(backbone, num_classes=len(species_to_index)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode="min")

        best_state_dict = None
        best_val_metrics = None
        best_epoch = 0
        epochs_trained = 0
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            epochs_trained = epoch + 1
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"]},
                step=epoch,
            )
            print(
                f"[{backbone}] epoch {epoch + 1}/{EPOCHS}: train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f}"
            )

            should_stop = early_stopping.step(val_metrics["loss"])
            if early_stopping.epochs_without_improvement == 0:
                # New best -- keep its weights (on CPU, so we're not holding two copies of the
                # model on GPU at once) so a later, worse epoch doesn't overwrite the checkpoint.
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_val_metrics = val_metrics
                best_epoch = epochs_trained

            if should_stop:
                print(
                    f"[{backbone}] stopping early after {epochs_trained} epochs -- val_loss hasn't "
                    f"improved for {EARLY_STOPPING_PATIENCE} consecutive epochs "
                    f"(best val_loss={early_stopping.best_score:.4f} at epoch {best_epoch})"
                )
                break

        model.load_state_dict(best_state_dict)
        results[backbone] = best_val_metrics
        mlflow.log_metrics({"final_val_accuracy": best_val_metrics["accuracy"]})
        mlflow.log_params({
            "epochs_trained": epochs_trained, "stopped_early": early_stopping.should_stop,
            "best_epoch": best_epoch,
        })

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = CHECKPOINT_DIR / f"{backbone}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        mlflow.log_artifact(str(checkpoint_path))

        species_mapping_path = CHECKPOINT_DIR / "species_to_index.json"
        with open(species_mapping_path, "w", encoding="utf-8") as f:
            json.dump(species_to_index, f, indent=2)
        mlflow.log_artifact(str(species_mapping_path))

print("\nComparison (final val accuracy):")
for backbone, metrics in results.items():
    print(f"  {backbone}: {metrics['accuracy']:.3f}")
