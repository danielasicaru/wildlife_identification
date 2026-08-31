from src.api.config import ServeConfig, load_serve_config


def test_load_serve_config_returns_dataclass(tmp_path):
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(
        "backbone: efficientnet_b0\n"
        "checkpoint_dir: data/checkpoints\n"
        "megadetector_model_name: MDV5A\n"
        "min_confidence: 0.2\n"
        "box_expansion_fraction: 0.1\n"
        "host: 127.0.0.1\n"
        "port: 8000\n"
    )

    config = load_serve_config(config_path)

    assert isinstance(config, ServeConfig)
    assert config.backbone == "efficientnet_b0"
    assert config.min_confidence == 0.2
