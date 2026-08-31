from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.config import ServeConfig


def _dummy_config() -> ServeConfig:
    return ServeConfig(
        backbone="efficientnet_b0", checkpoint_dir="unused", megadetector_model_name="unused",
        min_confidence=0.2, box_expansion_fraction=0.1, host="127.0.0.1", port=8000,
    )


def test_health_endpoint_does_not_require_state():
    app = create_app(_dummy_config(), state=MagicMock())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_detections(monkeypatch, tmp_path):
    from PIL import Image

    fake_state = MagicMock()
    app = create_app(_dummy_config(), state=fake_state)

    monkeypatch.setattr("src.api.app.predict", lambda image, state: [
        {"bbox": [1, 2, 3, 4], "species": "fox", "confidence": 0.9}
    ])

    image_path = tmp_path / "test.jpg"
    Image.new("RGB", (50, 50)).save(image_path)

    with TestClient(app) as client, open(image_path, "rb") as f:
        response = client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert body["detections"] == [{"bbox": [1, 2, 3, 4], "species": "fox", "confidence": 0.9}]


def test_create_app_uses_lifespan_when_no_state_given(monkeypatch):
    """Confirms the expensive loader is wired to lifespan, not called eagerly at create_app() time."""
    build_called = []
    monkeypatch.setattr("src.api.app.build_app_state", lambda config: build_called.append(config) or MagicMock())

    app = create_app(_dummy_config())  # no state passed -- production path

    assert build_called == []  # not called yet -- only on actual startup
    with TestClient(app):  # triggers lifespan startup
        pass
    assert len(build_called) == 1
