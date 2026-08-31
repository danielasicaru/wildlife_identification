"""Detect -> crop -> classify for a single uploaded image. Reuses the same functions
training/evaluation use for preprocessing, so serving can't silently drift from what the model
was actually trained/evaluated on."""
import torch
from PIL import Image

from src.api.state import AppState
from src.localization.crop import crop_to_bbox, expand_bbox
from src.localization.detector import bbox_to_absolute, filter_animal_detections, run_detection


def predict(image: Image.Image, state: AppState) -> list[dict]:
    raw_result = run_detection(state.detector, image, image_id="request")
    animal_detections = filter_animal_detections(raw_result["detections"], min_confidence=state.min_confidence)

    results = []
    for detection in animal_detections:
        bbox_abs = bbox_to_absolute(detection["bbox"], image.width, image.height)
        bbox_expanded = expand_bbox(bbox_abs, state.box_expansion_fraction, image.width, image.height)
        crop = crop_to_bbox(image, bbox_expanded).convert("RGB")

        tensor = state.val_transform(crop).unsqueeze(0).to(state.device)
        with torch.no_grad():
            logits = state.classifier(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_index = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, pred_index].item())

        results.append({
            "bbox": list(bbox_expanded),
            "species": state.index_to_species[pred_index],
            "confidence": confidence,
        })

    return results
