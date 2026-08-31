"""FastAPI app factory. Expensive resources (MegaDetector, classifier weights) are loaded once
in the lifespan hook and stored on app.state -- never at import time, never per-request."""
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile
from PIL import Image

from src.api.config import ServeConfig
from src.api.inference import predict
from src.api.state import AppState, build_app_state


def create_app(config: ServeConfig, state: AppState | None = None) -> FastAPI:
    """If `state` is given, it's attached directly and no lifespan loading happens -- this is
    what makes the app testable without loading real model weights. If `state` is None (the
    production path), a lifespan hook builds the real AppState from `config` once at startup.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.inference_state = state if state is not None else build_app_state(config)
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/predict")
    async def predict_endpoint(file: UploadFile):
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        detections = predict(image, app.state.inference_state)
        return {"detections": detections}

    return app
