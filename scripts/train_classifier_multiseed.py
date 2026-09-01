"""Trains and evaluates every backbone across multiple seeds, reseeding the train/val/test split
(not just model init and sampler order) each time -- so this captures split-sensitivity too, not
just training-randomness sensitivity, which matters at this dataset size. Turns the primary
single-seed backbone comparison into a mean +/- std comparison. Both trains and evaluates within
the same run (unlike the other scripts' train/evaluate split) since each of the N seeds x
backbones checkpoints needs evaluating immediately, not looked up later as "the latest batch."
See reports/multiseed_evaluation.md for the result."""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import numpy as np
import pandas as pd
import torch

from src.classifier.data_prep import build_labeled_crop_df
from src.classifier.models import build_model
from src.classifier.prediction import predict_test_set
from src.classifier.split import group_images_by_near_duplicates, split_groups
from src.classifier.training_run import train_and_compare_backbones
from src.data.quality import find_near_duplicates
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS_PATH = ROOT / "data" / "localization" / "detections.json"
CROPS_DIR = ROOT / "data" / "localization" / "crops"
ANNOTATIONS_PATH = ROOT / "data" / "raw" / "caltech_images_20210113.json"
BBOX_PATH = ROOT / "data" / "raw" / "caltech_bboxes_20200316.json"
IMAGES_DIR = ROOT / "data" / "raw" / "images"
NEAR_DUPLICATES_PATH = ROOT / "data" / "near_duplicates.json"
CONFIG_PATH = ROOT / "configs" / "train_classifier_multiseed.yaml"
CHECKPOINT_DIR = ROOT / "data" / "checkpoints_multiseed"
REPORT_PATH = ROOT / "reports" / "multiseed_evaluation.md"

config = load_config(CONFIG_PATH)
SEEDS = config["seeds"]
EPOCHS = config["epochs"]
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = config["learning_rate"]
MIN_SAMPLES_PER_SPECIES = config["min_samples_per_species"]
EARLY_STOPPING_PATIENCE = config["early_stopping_patience"]
BACKBONES = tuple(config["backbones"])

if EPOCHS < 1:
    raise SystemExit(f"configs/train_classifier_multiseed.yaml: epochs must be >= 1, got {EPOCHS}")

for path in (DETECTIONS_PATH, ANNOTATIONS_PATH, BBOX_PATH):
    if not path.exists():
        raise SystemExit(f"{path} not found -- run the localization and download scripts first.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if NEAR_DUPLICATES_PATH.exists():
    with open(NEAR_DUPLICATES_PATH, encoding="utf-8") as f:
        duplicate_pairs = [tuple(pair) for pair in json.load(f)]
else:
    sample_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    duplicate_pairs = [(a.name, b.name) for a, b in find_near_duplicates(sample_paths)]

mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
mlflow.set_experiment("camera-trap-classifier-multiseed")

results = {backbone: [] for backbone in BACKBONES}  # backbone -> list of (seed, test_accuracy)

for seed in SEEDS:
    print(f"\n=== seed {seed} ===")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    crop_df, _ = build_labeled_crop_df(DETECTIONS_PATH, ANNOTATIONS_PATH, BBOX_PATH, MIN_SAMPLES_PER_SPECIES)
    groups = group_images_by_near_duplicates(crop_df["source_image"].unique().tolist(), duplicate_pairs)
    crop_df["group_id"] = crop_df["source_image"].map(groups)
    crop_df["split"] = split_groups(crop_df, seed=seed)

    train_df = crop_df[crop_df["split"] == "train"].reset_index(drop=True)
    train_species = set(train_df["species"].unique())
    species_without_train_examples = set(crop_df["species"].unique()) - train_species
    if species_without_train_examples:
        print(f"  dropping {len(species_without_train_examples)} species with zero train examples: {sorted(species_without_train_examples)}")
        crop_df = crop_df[crop_df["species"].isin(train_species)].reset_index(drop=True)
        train_df = crop_df[crop_df["split"] == "train"].reset_index(drop=True)

    val_df = crop_df[crop_df["split"] == "val"].reset_index(drop=True)
    test_df = crop_df[crop_df["split"] == "test"].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        print(f"  skipping seed {seed}: an empty split")
        continue

    species_to_index = {s: i for i, s in enumerate(sorted(crop_df["species"].unique()))}
    index_to_species = {i: s for s, i in species_to_index.items()}

    backbone_val_results = train_and_compare_backbones(
        train_df, val_df, CROPS_DIR, species_to_index, BACKBONES, seed, EPOCHS, BATCH_SIZE,
        LEARNING_RATE, EARLY_STOPPING_PATIENCE, device, CHECKPOINT_DIR,
        mlflow_params={"comparison": "multiseed"}, artifact_paths=[CONFIG_PATH],
    )

    for backbone in BACKBONES:
        model = build_model(backbone, num_classes=len(species_to_index), pretrained=False).to(device)
        model.load_state_dict(torch.load(CHECKPOINT_DIR / f"{backbone}.pt", map_location=device))
        model.eval()

        predictions = predict_test_set(model, test_df, CROPS_DIR, species_to_index, index_to_species, device)
        test_accuracy = (predictions["true"] == predictions["predicted"]).mean()
        results[backbone].append((seed, test_accuracy))
        print(f"  [{backbone}] test_accuracy={test_accuracy:.3f} (val_accuracy={backbone_val_results[backbone]['accuracy']:.3f})")

# --- Aggregate ---
summary_rows = []
for backbone in BACKBONES:
    accs = pd.Series([acc for _, acc in results[backbone]])
    summary_rows.append({
        "backbone": backbone, "n_seeds": len(accs), "mean_test_accuracy": accs.mean(),
        "std_test_accuracy": accs.std(ddof=1) if len(accs) > 1 else 0.0,
        "min": accs.min(), "max": accs.max(),
    })
summary = pd.DataFrame(summary_rows).sort_values("mean_test_accuracy", ascending=False)

detail_rows = [
    {"backbone": backbone, "seed": seed, "test_accuracy": round(acc, 3)}
    for backbone in BACKBONES
    for seed, acc in results[backbone]
]
detail = pd.DataFrame(detail_rows).pivot(index="seed", columns="backbone", values="test_accuracy")

lines = [
    "# Multi-Seed Backbone Comparison",
    "",
    f"Each backbone trained and evaluated across {len(SEEDS)} seeds ({', '.join(str(s) for s in SEEDS)}), "
    "reseeding the train/val/test split (not just model init/sampler order) each time -- this "
    "captures split-sensitivity too, not just training-randomness, which matters at this dataset "
    "size. Numbers below are test accuracy per seed, not validation accuracy.",
    "",
    summary.round(3).to_markdown(index=False),
    "",
    "## Per-seed test accuracy",
    "",
    detail.to_markdown(),
]

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\nMean test accuracy by backbone:")
for _, row in summary.iterrows():
    print(f"  {row['backbone']}: {row['mean_test_accuracy']:.3f} +/- {row['std_test_accuracy']:.3f} (n={row['n_seeds']})")
print(f"Report: {REPORT_PATH}")
