"""Evaluates the best backbone from the most recent site-holdout training run (see
scripts/train_classifier_site_holdout.py) on its site-disjoint test split: overall + per-class
accuracy, so it can be compared against reports/classifier_evaluation.md's near-duplicate-split
number as a measure of unseen-site generalization. Writes reports/site_holdout_evaluation.md."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import torch

from src.classifier.data_prep import build_labeled_crop_df
from src.classifier.models import build_model
from src.classifier.prediction import predict_test_set
from src.classifier.split import group_images_by_site, split_groups
from src.evaluation.classifier_metrics import per_class_report
from src.evaluation.segmentation import build_site_lookup
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS_PATH = ROOT / "data" / "localization" / "detections.json"
CROPS_DIR = ROOT / "data" / "localization" / "crops"
ANNOTATIONS_PATH = ROOT / "data" / "raw" / "caltech_images_20210113.json"
BBOX_PATH = ROOT / "data" / "raw" / "caltech_bboxes_20200316.json"
REPORT_PATH = ROOT / "reports" / "site_holdout_evaluation.md"

train_config = load_config(ROOT / "configs" / "train_classifier_site_holdout.yaml")
SEED = train_config["seed"]
MIN_SAMPLES_PER_SPECIES = train_config["min_samples_per_species"]

crop_df, images_df = build_labeled_crop_df(DETECTIONS_PATH, ANNOTATIONS_PATH, BBOX_PATH, MIN_SAMPLES_PER_SPECIES)

site_lookup = build_site_lookup(images_df)
groups = group_images_by_site(crop_df["source_image"].unique().tolist(), site_lookup)
crop_df["group_id"] = crop_df["source_image"].map(groups)
crop_df["split"] = split_groups(crop_df, seed=SEED)

train_species = set(crop_df[crop_df["split"] == "train"]["species"].unique())
dropped_species = sorted(set(crop_df["species"].unique()) - train_species)
crop_df = crop_df[crop_df["species"].isin(train_species)].reset_index(drop=True)
test_df = crop_df[crop_df["split"] == "test"].reset_index(drop=True)

if test_df.empty:
    raise SystemExit("Test split is empty -- nothing to evaluate.")

mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
runs = mlflow.search_runs(experiment_names=["camera-trap-classifier-site-holdout"], order_by=["start_time DESC"])
if runs.empty:
    raise SystemExit("No MLflow runs found -- run scripts/train_classifier_site_holdout.py first.")

latest_batch = runs.head(len(train_config["backbones"]))
best_run = latest_batch.sort_values("metrics.final_val_accuracy", ascending=False).iloc[0]
backbone = best_run["tags.mlflow.runName"]
print(f"Evaluating best run from the latest site-holdout batch: {backbone} (val_accuracy={best_run['metrics.final_val_accuracy']:.3f})")

client = mlflow.tracking.MlflowClient()
artifact_dir = Path(client.download_artifacts(best_run["run_id"], ""))
species_to_index = json.loads((artifact_dir / "species_to_index.json").read_text(encoding="utf-8"))
index_to_species = {v: k for k, v in species_to_index.items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(backbone, num_classes=len(species_to_index), pretrained=False).to(device)
model.load_state_dict(torch.load(artifact_dir / f"{backbone}.pt", map_location=device))
model.eval()

predictions = predict_test_set(model, test_df, CROPS_DIR, species_to_index, index_to_species, device)
y_true = predictions["true"].tolist()
y_pred = predictions["predicted"].tolist()

labels = sorted(species_to_index.keys())
report = per_class_report(y_true, y_pred, labels)
overall_accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

train_sites = crop_df[crop_df["split"] == "train"]["source_image"].map(groups).nunique()
val_sites = crop_df[crop_df["split"] == "val"]["source_image"].map(groups).nunique()
test_sites = test_df["source_image"].map(groups).nunique()

lines = [
    "# Site-Holdout Generalization Check",
    "",
    "Same crops and labeling as the main classifier evaluation, but split so every crop from a "
    "given camera site lands entirely in train, val, or test -- no site appears in more than one "
    "split. Test accuracy here measures generalization to camera sites the model never saw during "
    "training, as opposed to reports/classifier_evaluation.md's split, which only guarantees "
    "unseen images (the same site can appear in both train and test there).",
    "",
    f"Best run (latest site-holdout batch): **{backbone}** (val_accuracy={best_run['metrics.final_val_accuracy']:.3f})",
    f"Test set: {len(test_df)} crops across {len(labels)} species, from {test_sites} camera sites "
    f"(train: {len(crop_df[crop_df['split'] == 'train'])} crops / {train_sites} sites, "
    f"val: {len(crop_df[crop_df['split'] == 'val'])} crops / {val_sites} sites).",
    f"Overall test accuracy: **{overall_accuracy:.1%}**",
    "",
]
if dropped_species:
    lines += [
        f"{len(dropped_species)} species dropped entirely -- confined to camera sites that all "
        f"landed outside the train split, so the model never saw a training example: "
        f"{', '.join(dropped_species)}.",
        "",
    ]
lines += [
    "## Per-class metrics",
    "",
    report.round(3).to_markdown(),
]

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Overall test accuracy: {overall_accuracy:.1%}")
print(f"Report: {REPORT_PATH}")
