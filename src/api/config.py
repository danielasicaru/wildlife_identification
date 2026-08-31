"""Serving configuration, loaded from configs/serve.yaml like every other script's config."""
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import load_config


@dataclass
class ServeConfig:
    backbone: str
    checkpoint_dir: str
    megadetector_model_name: str
    min_confidence: float
    box_expansion_fraction: float
    host: str
    port: int


def load_serve_config(path: str | Path) -> ServeConfig:
    return ServeConfig(**load_config(path))
