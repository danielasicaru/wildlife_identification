import json
from unittest.mock import MagicMock

from src.api.config import ServeConfig
from src.api.state import build_app_state


def test_build_app_state_wires_config_and_species_mapping(tmp_path, monkeypatch):
    species_index_path = tmp_path / "species_to_index.json"
    species_index_path.write_text(json.dumps({"fox": 0, "coyote": 1}))
    checkpoint_path = tmp_path / "efficientnet_b0.pt"
    checkpoint_path.write_bytes(b"fake checkpoint contents")

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model  # mirrors nn.Module.to(), which returns self
    monkeypatch.setattr("src.api.state.build_model", lambda *a, **k: fake_model)
    monkeypatch.setattr("src.api.state.torch.load", lambda *a, **k: {})
    monkeypatch.setattr("src.api.state.load_detector", lambda model_name: f"detector-for-{model_name}")

    fake_model_version = MagicMock(run_id="run123")
    fake_run = MagicMock()
    fake_run.data.params = {"backbone": "efficientnet_b0"}

    fake_client = MagicMock()
    fake_client.get_model_version_by_alias.return_value = fake_model_version
    fake_client.get_run.return_value = fake_run
    fake_client.download_artifacts.side_effect = lambda run_id, path: {
        "species_to_index.json": str(species_index_path),
        "efficientnet_b0.pt": str(checkpoint_path),
    }[path]
    monkeypatch.setattr("src.api.state.MlflowClient", lambda: fake_client)

    config = ServeConfig(
        registered_model_name="camera-trap-classifier", registered_model_alias="production",
        megadetector_model_name="MDV5A", min_confidence=0.3, box_expansion_fraction=0.15,
        host="127.0.0.1", port=8000, max_upload_bytes=10_485_760,
    )

    state = build_app_state(config)

    fake_client.get_model_version_by_alias.assert_called_once_with("camera-trap-classifier", "production")
    assert state.species_to_index == {"fox": 0, "coyote": 1}
    assert state.index_to_species == {0: "fox", 1: "coyote"}
    assert state.detector == "detector-for-MDV5A"
    assert state.min_confidence == 0.3
    assert state.box_expansion_fraction == 0.15
    fake_model.eval.assert_called_once()
    fake_model.load_state_dict.assert_called_once_with({})
