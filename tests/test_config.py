import pytest

from src.utils.config import load_config


def test_load_config_returns_dict_from_yaml(tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text("seed: 42\nlearning_rate: 0.0001\n")

    config = load_config(config_path)

    assert config == {"seed": 42, "learning_rate": 0.0001}


def test_load_config_raises_clear_error_when_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError, match="does_not_exist.yaml"):
        load_config(missing_path)
