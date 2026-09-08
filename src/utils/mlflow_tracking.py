"""Shared MLflow tracking URI. A SQLite-backed store, not the plain file store every script used
before -- the Model Registry (registration, versions, aliases) requires a database-backed
tracking store; the file store doesn't support it at all."""
from pathlib import Path


def mlflow_tracking_uri(root: Path) -> str:
    return f"sqlite:///{root / 'mlflow.db'}"
