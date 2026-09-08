from src.api.config import ServeConfig, load_serve_config


def test_load_serve_config_returns_dataclass(tmp_path):
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(
        "registered_model_name: camera-trap-classifier\n"
        "registered_model_alias: production\n"
        "megadetector_model_name: MDV5A\n"
        "min_confidence: 0.2\n"
        "box_expansion_fraction: 0.1\n"
        "host: 127.0.0.1\n"
        "port: 8000\n"
        "max_upload_bytes: 10485760\n"
    )

    config = load_serve_config(config_path)

    assert isinstance(config, ServeConfig)
    assert config.registered_model_name == "camera-trap-classifier"
    assert config.min_confidence == 0.2
