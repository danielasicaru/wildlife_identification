from pathlib import Path

from src.utils.mlflow_tracking import mlflow_tracking_uri


def test_mlflow_tracking_uri_is_sqlite_under_root():
    uri = mlflow_tracking_uri(Path("/repo"))

    assert uri.startswith("sqlite:///")
    assert uri.endswith("mlflow.db")
