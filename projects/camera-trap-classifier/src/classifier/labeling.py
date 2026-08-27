"""Per-crop species labeling, per ADR 0006: ground-truth IoU matching where available, single-
species fallback otherwise, drop when genuinely ambiguous."""
import pandas as pd

from src.localization.evaluate import iou


def match_crop_to_ground_truth(
    crop_bbox: tuple[int, int, int, int],
    ground_truth: list[tuple[tuple[int, int, int, int], str]],
) -> str | None:
    """Best-IoU-matching ground-truth species for a crop, or None if nothing overlaps at all."""
    best_species = None
    best_iou = 0.0
    for gt_bbox, species in ground_truth:
        score = iou(crop_bbox, gt_bbox)
        if score > best_iou:
            best_iou = score
            best_species = species
    return best_species


def build_crop_dataframe(
    detections: list[dict],
    image_species: dict[str, set[str]],
    image_ground_truth: dict[str, list[tuple[tuple[int, int, int, int], str]]],
) -> pd.DataFrame:
    """Assigns a species label to each crop, per ADR 0006's three-tier rule. Returns a DataFrame
    with columns [crop_file, source_image, species]; ambiguous/unmatched crops are excluded.
    """
    rows = []
    for result in detections:
        source_image = result["source_image"]
        species_set = image_species.get(source_image, set())
        ground_truth = image_ground_truth.get(source_image)

        for crop in result["crops"]:
            if ground_truth:
                species = match_crop_to_ground_truth(tuple(crop["bbox_detected"]), ground_truth)
            elif len(species_set) == 1:
                species = next(iter(species_set))
            else:
                species = None

            if species is not None:
                rows.append({"crop_file": crop["crop_file"], "source_image": source_image, "species": species})

    return pd.DataFrame(rows, columns=["crop_file", "source_image", "species"])
