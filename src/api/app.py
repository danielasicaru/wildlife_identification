"""FastAPI app factory. Expensive resources (MegaDetector, classifier weights) are loaded once
in the lifespan hook and stored on app.state -- never at import time, never per-request."""
import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.api.config import ServeConfig
from src.api.inference import predict
from src.api.state import AppState, build_app_state

logger = logging.getLogger(__name__)


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

    # Plain `def`, not `async def`: Starlette runs sync route handlers in a worker thread pool,
    # so the blocking MegaDetector/classifier forward passes below don't stall the event loop
    # (and everything else, e.g. /health) for other concurrent requests.
    @app.post("/predict")
    def predict_endpoint(file: UploadFile):
        image_bytes = file.file.read()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.")

        try:
            detections = predict(image, app.state.inference_state)
        except Exception:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail="Inference failed.")

        return {"detections": detections}

    return app
