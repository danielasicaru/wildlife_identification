"""FastAPI app factory. Expensive resources (MegaDetector, classifier weights) are loaded once
in the lifespan hook and stored on app.state -- never at import time, never per-request."""
import io
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from src.api.config import ServeConfig
from src.api.inference import predict
from src.api.metrics import RequestMetrics
from src.api.schemas import BatchPredictResponse, PredictResponse
from src.api.state import AppState, build_app_state

logger = logging.getLogger(__name__)


@dataclass
class DecodeError:
    status_code: int
    detail: str


def _decode_upload(file: UploadFile, max_bytes: int) -> tuple[Image.Image | None, DecodeError | None]:
    """Reads at most max_bytes+1 bytes (never buffers more, regardless of how large the actual
    upload is) and decodes it. Returns (image, None) on success, or (None, DecodeError) --
    callers decide whether to raise (single-image endpoint) or collect and continue (batch).
    """
    image_bytes = file.file.read(max_bytes + 1)
    if len(image_bytes) > max_bytes:
        return None, DecodeError(413, f"Upload exceeds the {max_bytes}-byte limit.")
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB"), None
    except UnidentifiedImageError:
        return None, DecodeError(400, "Uploaded file is not a readable image.")


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
    app.state.metrics = RequestMetrics()

    @app.middleware("http")
    async def log_and_record_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        app.state.metrics.record(response.status_code, duration_ms)
        logger.info(
            f"method={request.method} path={request.url.path} "
            f"status_code={response.status_code} duration_ms={duration_ms:.1f}"
        )
        return response

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return app.state.metrics.summary()

    # Plain `def`, not `async def`: Starlette runs sync route handlers in a worker thread pool,
    # so the blocking MegaDetector/classifier forward passes below don't stall the event loop
    # (and everything else, e.g. /health) for other concurrent requests.
    @app.post("/predict", response_model=PredictResponse)
    def predict_endpoint(file: UploadFile):
        image, decode_error = _decode_upload(file, config.max_upload_bytes)
        if decode_error is not None:
            raise HTTPException(status_code=decode_error.status_code, detail=decode_error.detail)

        try:
            detections = predict(image, app.state.inference_state)
        except Exception:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail="Inference failed.")

        return {"detections": detections}

    # Batch variant of /predict: reuses the same loaded AppState (detector + classifier stay in
    # memory, no per-file reload), amortizing MegaDetector's per-call overhead across a burst of
    # images -- the realistic camera-trap serving pattern, vs. one-image-at-a-time. A single bad
    # file doesn't fail the whole batch; its result entry carries an "error" instead of
    # "detections", the rest of the batch still gets scored.
    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch_endpoint(files: list[UploadFile]):
        results = []
        for file in files:
            image, decode_error = _decode_upload(file, config.max_upload_bytes)
            if decode_error is not None:
                results.append({"filename": file.filename, "error": decode_error.detail})
                continue

            try:
                detections = predict(image, app.state.inference_state)
            except Exception:
                logger.exception(f"Inference failed for {file.filename}")
                results.append({"filename": file.filename, "error": "Inference failed."})
                continue

            results.append({"filename": file.filename, "detections": detections})

        return {"results": results}

    return app
