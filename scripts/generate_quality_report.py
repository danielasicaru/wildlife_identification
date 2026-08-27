"""Runs quality checks on the downloaded sample images and writes a markdown report with
findings and recommended follow-up actions -- not just raw numbers.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.characterize import metadata_survey
from src.data.loader import load_annotations
from src.data.quality import blur_score, find_near_duplicates, is_effectively_grayscale, is_valid_image

ANNOTATIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "caltech_images_20210113.json"
IMAGES_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "images"
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "quality.md"
# Persisted so other scripts (e.g. train_classifier.py) don't have to recompute this O(n^2)
# perceptual-hash comparison from scratch.
NEAR_DUPLICATES_PATH = Path(__file__).resolve().parents[1] / "data" / "near_duplicates.json"

# Below this Laplacian-variance score, an image is flagged as a blur candidate. Calibrated as the
# 10th percentile of this sample's own score distribution, not an arbitrary fixed number, since
# "sharp" varies a lot by camera/lighting conditions across a real dataset.
BLUR_PERCENTILE = 10


def main() -> None:
    paths = sorted(IMAGES_DIR.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"No images found in {IMAGES_DIR} -- run download_sample_images.py first.")

    corrupted = [p for p in paths if not is_valid_image(p)]
    corrupted_set = set(corrupted)
    valid_paths = [p for p in paths if p not in corrupted_set]

    if not valid_paths:
        raise SystemExit(
            f"All {len(paths)} sampled images failed quality checks (corrupted or undecodable) -- "
            "nothing to analyze."
        )

    # Computed directly from the annotations, not copied from characterization.md, so this stays
    # correct if the annotations file or the day/night proxy logic ever changes.
    all_images, _, _ = load_annotations(ANNOTATIONS_PATH)
    hour_proxy_meta = metadata_survey(all_images)
    hour_proxy_night_share = hour_proxy_meta["night_count"] / (
        hour_proxy_meta["day_count"] + hour_proxy_meta["night_count"]
    )

    scores = {p: blur_score(p) for p in valid_paths}
    sorted_scores = sorted(scores.values())
    blur_threshold = statistics.quantiles(sorted_scores, n=100, method="inclusive")[BLUR_PERCENTILE - 1]
    blurry = [p for p, s in scores.items() if s <= blur_threshold]

    grayscale_flags = {p: is_effectively_grayscale(p) for p in valid_paths}
    n_gray = sum(grayscale_flags.values())
    gray_share = n_gray / len(valid_paths)

    duplicates = find_near_duplicates(valid_paths)

    NEAR_DUPLICATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NEAR_DUPLICATES_PATH, "w", encoding="utf-8") as f:
        json.dump([[a.name, b.name] for a, b in duplicates], f, indent=2)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Quality Checks — Caltech Camera Traps (sample)\n\n")
        f.write(f"Sample size: {len(paths)} images (stratified, ~20 per category)\n\n")

        f.write("## Corruption\n\n")
        f.write(f"Corrupted: {len(corrupted)}/{len(paths)}\n\n")
        if corrupted:
            f.write("Files: " + ", ".join(p.name for p in corrupted) + "\n\n")

        f.write("## Blur\n\n")
        f.write(
            f"Laplacian variance: min={min(sorted_scores):.1f}, max={max(sorted_scores):.1f}, "
            f"median={statistics.median(sorted_scores):.1f}\n\n"
        )
        f.write(f"Flagged as blur candidates (bottom {BLUR_PERCENTILE}%, threshold={blur_threshold:.1f}): "
                f"{len(blurry)}/{len(valid_paths)}\n\n")

        f.write("## Color channel / day-night proxy validation\n\n")
        f.write(f"Effectively grayscale (real pixel check, tolerance=2.0): "
                f"{n_gray}/{len(valid_paths)} ({100*gray_share:.1f}%)\n\n")
        f.write(f"Hour-based day/night proxy (computed from annotations) estimated "
                f"{100*hour_proxy_night_share:.1f}% night.\n\n")
        gap = abs(gray_share - hour_proxy_night_share)
        f.write(f"Gap between pixel-based and hour-based estimates: {100*gap:.1f} percentage points.\n\n")

        f.write("## Near-duplicates\n\n")
        f.write(f"Near-duplicate pairs found (perceptual hash, Hamming distance <= 5): {len(duplicates)}\n\n")

        f.write("## Findings and recommended follow-up\n\n")
        f.write(
            "- **Corruption**: "
            + (
                "none found in this sample. Not a concern for this dataset at this sample size; "
                "worth a full-dataset pass before final training, not before."
                if not corrupted
                else f"{len(corrupted)} corrupted/undecodable file(s) found in this sample -- "
                "recommend a full-dataset corruption pass before training, and excluding these "
                "specific files from any split."
            )
            + "\n"
        )
        f.write(
            "- **Blur**: real spread exists (see min/max above). Recommend visually spot-checking "
            "the flagged bottom-decile images before deciding whether to exclude them or keep them "
            "as intentionally hard examples -- camera traps trigger on motion, so some blur may be "
            "an inherent, unavoidable part of the true data distribution, not noise to remove.\n"
        )
        f.write(
            f"- **Day/night proxy accuracy**: the {100*gap:.0f}-point gap between the pixel-based "
            "grayscale check and the hour-based proxy means the hour proxy is not reliable enough "
            "to use as a ground-truth label. Recommend replacing the proxy with the pixel-based "
            "`is_effectively_grayscale` check directly wherever day/night matters downstream (e.g. "
            "for the selective grayscale augmentation's targeting logic), rather than widening the "
            "day/night hour window to try to match it.\n"
        )
        f.write(
            "- **Near-duplicates**: "
            + (
                "none found in this sample; camera trap burst sequences (seq_num_frames) may still "
                "produce near-duplicates at full-dataset scale even if absent here given the small "
                "per-class sample size."
                if not duplicates
                else "found in this sample -- recommend deduplicating before creating train/val/test "
                "splits, so near-identical frames don't leak across splits and inflate validation "
                "metrics."
            )
            + "\n"
        )

    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
