"""Small grid search over learning rate and batch size, on the best backbone only
(EfficientNet-B0), one seed each -- the multi-seed comparison already showed backbone choice
barely matters at this dataset size, so this asks a narrower question: given a fixed backbone
and split, was the originally-chosen learning_rate/batch_size in configs/train_classifier.yaml
actually a good pick, or just the first thing tried. Same split (seed fixed across every grid
point) so only the hyperparameters vary, not the data. See reports/hyperparameter_search.md."""
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
from src.classifier.split import group_images_by_near_duplicates, split_groups
from src.classifier.training_run import train_and_compare_backbones
from src.data.quality import find_near_duplicates
from src.utils.config import load_config
from src.utils.mlflow_tracking import mlflow_tracking_uri

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS_PATH = ROOT / "data" / "localization" / "detections.json"
CROPS_DIR = ROOT / "data" / "localization" / "crops"
ANNOTATIONS_PATH = ROOT / "data" / "raw" / "caltech_images_20210113.json"
BBOX_PATH = ROOT / "data" / "raw" / "caltech_bboxes_20200316.json"
IMAGES_DIR = ROOT / "data" / "raw" / "images"
NEAR_DUPLICATES_PATH = ROOT / "data" / "near_duplicates.json"
CONFIG_PATH = ROOT / "configs" / "hyperparameter_search.yaml"
CHECKPOINT_DIR = ROOT / "data" / "checkpoints_hp_search"
REPORT_PATH = ROOT / "reports" / "hyperparameter_search.md"

config = load_config(CONFIG_PATH)
SEED = config["seed"]
EPOCHS = config["epochs"]
EARLY_STOPPING_PATIENCE = config["early_stopping_patience"]
MIN_SAMPLES_PER_SPECIES = config["min_samples_per_species"]
BACKBONE = config["backbone"]
LEARNING_RATES = config["learning_rates"]
BATCH_SIZES = config["batch_sizes"]

for path in (DETECTIONS_PATH, ANNOTATIONS_PATH, BBOX_PATH):
    if not path.exists():
        raise SystemExit(f"{path} not found -- run the localization and download scripts first.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

crop_df, _ = build_labeled_crop_df(DETECTIONS_PATH, ANNOTATIONS_PATH, BBOX_PATH, MIN_SAMPLES_PER_SPECIES)

if NEAR_DUPLICATES_PATH.exists():
    with open(NEAR_DUPLICATES_PATH, encoding="utf-8") as f:
        duplicate_pairs = [tuple(pair) for pair in json.load(f)]
else:
    sample_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    duplicate_pairs = [(a.name, b.name) for a, b in find_near_duplicates(sample_paths)]
groups = group_images_by_near_duplicates(crop_df["source_image"].unique().tolist(), duplicate_pairs)
crop_df["group_id"] = crop_df["source_image"].map(groups)
crop_df["split"] = split_groups(crop_df, seed=SEED)

train_df = crop_df[crop_df["split"] == "train"].reset_index(drop=True)
train_species = set(train_df["species"].unique())
crop_df = crop_df[crop_df["species"].isin(train_species)].reset_index(drop=True)
train_df = crop_df[crop_df["split"] == "train"].reset_index(drop=True)
val_df = crop_df[crop_df["split"] == "val"].reset_index(drop=True)
print(f"train={len(train_df)}, val={len(val_df)}")

species_to_index = {s: i for i, s in enumerate(sorted(crop_df["species"].unique()))}

mlflow.set_tracking_uri(mlflow_tracking_uri(ROOT))
mlflow.set_experiment("camera-trap-classifier-hp-search")

grid_results = []
for learning_rate in LEARNING_RATES:
    for batch_size in BATCH_SIZES:
        print(f"\n=== learning_rate={learning_rate}, batch_size={batch_size} ===")
        results = train_and_compare_backbones(
            train_df, val_df, CROPS_DIR, species_to_index, (BACKBONE,), SEED, EPOCHS, batch_size,
            learning_rate, EARLY_STOPPING_PATIENCE, device, CHECKPOINT_DIR,
            mlflow_params={"comparison": "hp_search"}, artifact_paths=[CONFIG_PATH],
        )
        best_val_metrics = results[BACKBONE]
        grid_results.append({
            "learning_rate": learning_rate, "batch_size": batch_size,
            "val_accuracy": best_val_metrics["accuracy"], "val_loss": best_val_metrics["loss"],
        })

grid_df = pd.DataFrame(grid_results).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
# round(3) alone would display every one of these learning rates (5e-05, 1e-4, 5e-4) as 0.0 --
# round val_accuracy/val_loss but leave learning_rate at full precision.
grid_df["val_accuracy"] = grid_df["val_accuracy"].round(3)
grid_df["val_loss"] = grid_df["val_loss"].round(3)

lines = [
    "# Hyperparameter Search",
    "",
    f"Small grid search over learning rate and batch size on **{BACKBONE}** only, one seed "
    f"({SEED}) each, same train/val split for every grid point (only the hyperparameters vary). "
    "The multi-seed backbone comparison already showed backbone choice barely matters at this "
    "dataset size, so this asks a narrower, cheaper question: was the value already in "
    "`configs/train_classifier.yaml` a good pick, or just the first thing tried.",
    "",
    grid_df.to_markdown(index=False),
    "",
]

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\nGrid results (best first):")
print(grid_df.round(3).to_string(index=False))
print(f"Report: {REPORT_PATH}")
