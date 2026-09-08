"""Holds the resources loaded once at startup -- never rebuilt per-request, never loaded at
import time. Attached to app.state by the lifespan hook in app.py (or injected directly in
tests, bypassing the expensive real loader)."""
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from mlflow.tracking import MlflowClient
from torchvision.transforms import v2

from src.api.config import ServeConfig
from src.classifier.models import build_model
from src.data.augmentation import build_val_transform
from src.localization.detector import load_detector


@dataclass
class AppState:
    detector: object  # megadetector.detection.pytorch_detector.PTDetector
    classifier: torch.nn.Module
    species_to_index: dict[str, int]
    index_to_species: dict[int, str]
    device: str
    val_transform: v2.Compose
    min_confidence: float
    box_expansion_fraction: float


def build_app_state(config: ServeConfig) -> AppState:
    """The expensive path -- downloads/loads MegaDetector and the classifier checkpoint. Called
    once from the lifespan hook, never per-request.

    The classifier checkpoint is resolved via the MLflow Model Registry's `registered_model_alias`
    (e.g. "production") rather than a local file path + hardcoded backbone name -- this is what
    makes model promotion (scripts/promote_classifier.py) actually take effect here without
    editing this config. The registry only resolves *which run* is promoted; the actual weights
    are still loaded from that run's raw state_dict artifact with weights_only=True, not from the
    registry's own (pickle-serialized) logged model object -- see promote_classifier.py's
    docstring for why.
    """
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(config.registered_model_name, config.registered_model_alias)
    run = client.get_run(model_version.run_id)
    backbone = run.data.params["backbone"]

    species_index_path = client.download_artifacts(model_version.run_id, "species_to_index.json")
    species_to_index = json.loads(Path(species_index_path).read_text(encoding="utf-8"))
    index_to_species = {v: k for k, v in species_to_index.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = build_model(backbone, num_classes=len(species_to_index), pretrained=False).to(device)
    checkpoint_path = client.download_artifacts(model_version.run_id, f"{backbone}.pt")
    classifier.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    classifier.eval()

    detector = load_detector(config.megadetector_model_name)

    return AppState(
        detector=detector,
        classifier=classifier,
        species_to_index=species_to_index,
        index_to_species=index_to_species,
        device=device,
        val_transform=build_val_transform(),
        min_confidence=config.min_confidence,
        box_expansion_fraction=config.box_expansion_fraction,
    )
