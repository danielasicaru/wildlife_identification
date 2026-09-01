"""Response schemas for /predict -- gives FastAPI real OpenAPI docs at /docs instead of an
untyped dict, and validates the shape of what inference.py actually returns."""
from pydantic import BaseModel


class Detection(BaseModel):
    bbox: list[int]
    species: str
    confidence: float


class PredictResponse(BaseModel):
    detections: list[Detection]


class BatchDetectionResult(BaseModel):
    filename: str
    detections: list[Detection] | None = None
    error: str | None = None


class BatchPredictResponse(BaseModel):
    results: list[BatchDetectionResult]
