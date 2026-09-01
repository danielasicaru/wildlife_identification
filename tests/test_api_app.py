from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.config import ServeConfig


def _dummy_config(max_upload_bytes: int = 10_485_760) -> ServeConfig:
    return ServeConfig(
        backbone="efficientnet_b0", checkpoint_dir="unused", megadetector_model_name="unused",
        min_confidence=0.2, box_expansion_fraction=0.1, host="127.0.0.1", port=8000,
        max_upload_bytes=max_upload_bytes,
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


def test_predict_endpoint_rejects_non_image_upload():
    app = create_app(_dummy_config(), state=MagicMock())

    with TestClient(app) as client:
        response = client.post("/predict", files={"file": ("not_an_image.txt", b"hello world", "text/plain")})

    assert response.status_code == 400


def test_predict_endpoint_rejects_upload_over_size_limit(tmp_path):
    app = create_app(_dummy_config(max_upload_bytes=100), state=MagicMock())

    oversized_path = tmp_path / "big.jpg"
    oversized_path.write_bytes(b"x" * 200)

    with TestClient(app) as client, open(oversized_path, "rb") as f:
        response = client.post("/predict", files={"file": ("big.jpg", f, "image/jpeg")})

    assert response.status_code == 413


def test_predict_endpoint_returns_500_when_inference_raises(monkeypatch, tmp_path):
    from PIL import Image

    app = create_app(_dummy_config(), state=MagicMock())

    def raise_error(image, state):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.api.app.predict", raise_error)

    image_path = tmp_path / "test.jpg"
    Image.new("RGB", (50, 50)).save(image_path)

    with TestClient(app) as client, open(image_path, "rb") as f:
        response = client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})

    assert response.status_code == 500


def test_predict_batch_endpoint_returns_one_result_per_file(monkeypatch, tmp_path):
    from PIL import Image

    app = create_app(_dummy_config(), state=MagicMock())
    monkeypatch.setattr("src.api.app.predict", lambda image, state: [
        {"bbox": [1, 2, 3, 4], "species": "fox", "confidence": 0.9}
    ])

    path_a = tmp_path / "a.jpg"
    path_b = tmp_path / "b.jpg"
    Image.new("RGB", (50, 50)).save(path_a)
    Image.new("RGB", (50, 50)).save(path_b)

    with TestClient(app) as client, open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        response = client.post(
            "/predict/batch",
            files=[("files", ("a.jpg", fa, "image/jpeg")), ("files", ("b.jpg", fb, "image/jpeg"))],
        )

    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["filename"] for r in results] == ["a.jpg", "b.jpg"]
    assert all(r["detections"] == [{"bbox": [1, 2, 3, 4], "species": "fox", "confidence": 0.9}] for r in results)
    assert all(r["error"] is None for r in results)


def test_predict_batch_endpoint_reports_per_file_errors_without_failing_the_batch(monkeypatch, tmp_path):
    from PIL import Image

    app = create_app(_dummy_config(), state=MagicMock())
    monkeypatch.setattr("src.api.app.predict", lambda image, state: [])

    good_path = tmp_path / "good.jpg"
    Image.new("RGB", (50, 50)).save(good_path)

    with TestClient(app) as client, open(good_path, "rb") as good_file:
        response = client.post(
            "/predict/batch",
            files=[
                ("files", ("bad.txt", b"not an image", "text/plain")),
                ("files", ("good.jpg", good_file, "image/jpeg")),
            ],
        )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["filename"] == "bad.txt"
    assert results[0]["error"] == "Uploaded file is not a readable image."
    assert results[0]["detections"] is None
    assert results[1]["filename"] == "good.jpg"
    assert results[1]["detections"] == []
    assert results[1]["error"] is None


def test_metrics_endpoint_reflects_recorded_requests():
    app = create_app(_dummy_config(), state=MagicMock())

    with TestClient(app) as client:
        client.get("/health")
        client.get("/health")
        summary = client.get("/metrics").json()

    # 2, not 3: the /metrics call's own request is recorded by the middleware only *after*
    # call_next returns the response body (which the /metrics handler already built), so a
    # request never sees itself counted in its own response.
    assert summary["request_count"] == 2
    assert summary["error_count"] == 0
    assert summary["avg_latency_ms"] is not None


def test_predict_response_schema_is_documented_in_openapi():
    app = create_app(_dummy_config(), state=MagicMock())

    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    assert "PredictResponse" in schema["components"]["schemas"]
    assert "Detection" in schema["components"]["schemas"]


def test_create_app_uses_lifespan_when_no_state_given(monkeypatch):
    """Confirms the expensive loader is wired to lifespan, not called eagerly at create_app() time."""
    build_called = []
    monkeypatch.setattr("src.api.app.build_app_state", lambda config: build_called.append(config) or MagicMock())

    app = create_app(_dummy_config())  # no state passed -- production path

    assert build_called == []  # not called yet -- only on actual startup
    with TestClient(app):  # triggers lifespan startup
        pass
    assert len(build_called) == 1
