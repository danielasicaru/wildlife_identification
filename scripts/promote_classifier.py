"""Explicit model promotion: picks the best backbone from the most recent training batch (same
selection as evaluate_classifier.py) and registers it in the MLflow Model Registry under the
"production" alias. Replaces "evaluate_classifier.py always re-picks best-of-latest-batch" with
an explicit "this is the model currently serving in production" pointer that only changes when
this script is run -- src/api/state.py resolves the classifier to load via this alias, not by
re-running the best-of-latest-batch selection at serve startup.

The registered model object itself is logged in pickle format purely so MLflow's Model Registry
has something versionable to attach the alias to; serving does NOT load that pickled object. It
downloads the same raw state_dict artifact this run already logged (via
train_and_compare_backbones) and loads it with torch.load(..., weights_only=True), same as
evaluate_classifier.py -- the registry only resolves *which run* is production, it doesn't change
how the weights actually get loaded.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient

from src.classifier.models import BACKBONES, build_model
from src.utils.mlflow_tracking import mlflow_tracking_uri

ROOT = Path(__file__).resolve().parents[1]
REGISTERED_MODEL_NAME = "camera-trap-classifier"
ALIAS = "production"

mlflow.set_tracking_uri(mlflow_tracking_uri(ROOT))
client = MlflowClient()

runs = mlflow.search_runs(experiment_names=["camera-trap-classifier"], order_by=["start_time DESC"])
if runs.empty:
    raise SystemExit("No MLflow runs found -- run scripts/train_classifier.py first.")

latest_batch = runs.head(len(BACKBONES))
best_run = latest_batch.sort_values("metrics.final_val_accuracy", ascending=False).iloc[0]
run_id = best_run["run_id"]
backbone = best_run["tags.mlflow.runName"]
val_accuracy = best_run["metrics.final_val_accuracy"]
print(f"Promoting best run from the latest batch: {backbone} (val_accuracy={val_accuracy:.3f}, run_id={run_id})")

artifact_dir = Path(client.download_artifacts(run_id, ""))
species_to_index = json.loads((artifact_dir / "species_to_index.json").read_text(encoding="utf-8"))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(backbone, num_classes=len(species_to_index), pretrained=False).to(device)
model.load_state_dict(
    torch.load(artifact_dir / f"{backbone}.pt", map_location=device, weights_only=True)
)
model.eval()

with mlflow.start_run(run_id=run_id):
    model_info = mlflow.pytorch.log_model(
        model, name="model", registered_model_name=REGISTERED_MODEL_NAME, serialization_format="pickle",
    )

client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS, model_info.registered_model_version)
print(
    f"Registered '{REGISTERED_MODEL_NAME}' version {model_info.registered_model_version}, "
    f"aliased '{ALIAS}' -> backbone={backbone}, run_id={run_id}, val_accuracy={val_accuracy:.3f}"
)
