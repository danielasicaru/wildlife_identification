import json
from unittest.mock import MagicMock

from src.api.config import ServeConfig
from src.api.state import build_app_state


def test_build_app_state_wires_config_and_species_mapping(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "species_to_index.json").write_text(json.dumps({"fox": 0, "coyote": 1}))
    (checkpoint_dir / "efficientnet_b0.pt").write_bytes(b"fake checkpoint contents")

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model  # mirrors nn.Module.to(), which returns self
    monkeypatch.setattr("src.api.state.build_model", lambda *a, **k: fake_model)
    monkeypatch.setattr("src.api.state.torch.load", lambda *a, **k: {})
    monkeypatch.setattr("src.api.state.load_detector", lambda model_name: f"detector-for-{model_name}")

    config = ServeConfig(
        backbone="efficientnet_b0", checkpoint_dir=str(checkpoint_dir),
        megadetector_model_name="MDV5A", min_confidence=0.3, box_expansion_fraction=0.15,
        host="127.0.0.1", port=8000,
    )

    state = build_app_state(config)

    assert state.species_to_index == {"fox": 0, "coyote": 1}
    assert state.index_to_species == {0: "fox", 1: "coyote"}
    assert state.detector == "detector-for-MDV5A"
    assert state.min_confidence == 0.3
    assert state.box_expansion_fraction == 0.15
    fake_model.eval.assert_called_once()
    fake_model.load_state_dict.assert_called_once_with({})
