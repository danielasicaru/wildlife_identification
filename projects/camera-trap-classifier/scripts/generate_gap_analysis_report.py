"""Qualitative gap analysis: scale variation (quantitative, from bboxes) and occlusion (from
manually recorded tags, if present -- see notebooks/eda.ipynb for the tagging tool).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.characterize import bbox_area_ratio

BBOX_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "caltech_bboxes_20200316.json"
OCCLUSION_TAGS_PATH = Path(__file__).resolve().parents[1] / "data" / "occlusion_tags.json"
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "gap_analysis.md"


def main() -> None:
    if not BBOX_PATH.exists():
        raise SystemExit(f"No bounding-box annotations found at {BBOX_PATH} -- download "
                          "caltech_bboxes_20200316.json first (see project README).")

    with open(BBOX_PATH) as f:
        bbox_data = json.load(f)
    bboxes = pd.DataFrame(bbox_data["annotations"])
    images = pd.DataFrame(bbox_data["images"])

    ratios = bbox_area_ratio(bboxes, images)["area_ratio"]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Qualitative Gap Analysis — Scale Variation and Occlusion\n\n")

        f.write("## Scale variation\n\n")
        f.write(f"Annotated bounding boxes: {len(ratios)} (covering {images['id'].nunique()} of the "
                f"243,100 total images -- bbox coverage is partial, not the full dataset)\n\n")
        f.write(
            f"Animal-area-to-image-area ratio: min={ratios.min():.4f}, "
            f"25th={ratios.quantile(0.25):.4f}, median={ratios.median():.4f}, "
            f"75th={ratios.quantile(0.75):.4f}, max={ratios.max():.4f}\n\n"
        )
        f.write(f"Under 2% of frame area: {(ratios < 0.02).sum()}/{len(ratios)} "
                f"({100*(ratios < 0.02).mean():.1f}%)\n\n")
        f.write(f"Over 50% of frame area: {(ratios > 0.5).sum()}/{len(ratios)} "
                f"({100*(ratios > 0.5).mean():.1f}%)\n\n")
        f.write(
            "**Finding**: the median animal occupies under 3% of the frame, with a third of "
            "annotated images under 2%. This is a more extreme long tail than a qualitative "
            "'animals appear at different distances' statement suggests. **Follow-up**: revisit "
            "whether the augmentation spec's multi-scale crop range (0.7-1.0) adequately covers "
            "this tail, or whether a wider zoom-in range is needed to represent small, distant "
            "animals during training.\n\n"
        )

        f.write("## Occlusion\n\n")
        if OCCLUSION_TAGS_PATH.exists():
            with open(OCCLUSION_TAGS_PATH) as tf:
                tags = json.load(tf)
            counts = pd.Series(tags.values()).value_counts()
            f.write(f"Manually tagged sample: {len(tags)} images\n\n")
            for tag, count in counts.items():
                f.write(f"- {tag}: {count} ({100*count/len(tags):.1f}%)\n")
        else:
            f.write(
                "Not yet tagged. Occlusion has no reliable automatic proxy without segmentation "
                f"masks (see ADR 0002-style reasoning) -- run the tagging tool in "
                "`notebooks/eda.ipynb` and save results to `data/occlusion_tags.json`, then "
                "re-run this script.\n"
            )

    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
