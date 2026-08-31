"""Holds the resources loaded once at startup -- never rebuilt per-request, never loaded at
import time. Attached to app.state by the lifespan hook in app.py (or injected directly in
tests, bypassing the expensive real loader)."""
import json
from dataclasses import dataclass
from pathlib import Path

import torch
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
    """
    checkpoint_dir = Path(config.checkpoint_dir)
    species_to_index = json.loads((checkpoint_dir / "species_to_index.json").read_text(encoding="utf-8"))
    index_to_species = {v: k for k, v in species_to_index.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = build_model(config.backbone, num_classes=len(species_to_index), pretrained=False).to(device)
    classifier.load_state_dict(torch.load(checkpoint_dir / f"{config.backbone}.pt", map_location=device))
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
