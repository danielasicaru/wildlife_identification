"""Runs dataset characterization on the real CCT annotations and writes a markdown report."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_annotations
from src.data.characterize import class_distribution, metadata_survey, per_site_distribution

NON_SPECIES = {"empty", "car"}
ANNOTATIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "caltech_images_20210113.json"
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "characterization.md"


def main() -> None:
    images, annotations, categories = load_annotations(ANNOTATIONS_PATH)

    classes = class_distribution(annotations, categories, exclude=NON_SPECIES)
    sites = per_site_distribution(images, annotations, categories, exclude=NON_SPECIES)
    meta = metadata_survey(images)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Dataset Characterization — Caltech Camera Traps\n\n")
        f.write(f"Total images: {len(images)}\n\n")
        f.write(f"Total sites: {images['location'].nunique()}\n\n")

        f.write("## Class distribution\n\n")
        f.write("| Species | Count | Imbalance ratio |\n|---|---|---|\n")
        for name, row in classes.iterrows():
            f.write(f"| {name} | {int(row['count'])} | {row['imbalance_ratio']:.1f}x |\n")

        f.write("\n## Metadata survey\n\n")
        f.write(f"Day-proxy count: {meta['day_count']}\n\n")
        f.write(f"Night-proxy count: {meta['night_count']}\n\n")
        f.write(f"Unparseable date_captured count: {meta['unparseable_date_count']}\n\n")
        f.write("Resolution counts:\n\n")
        for (h, w), count in meta["resolution_counts"].items():
            f.write(f"- {w}x{h}: {count}\n")

        f.write(f"\n## Per-site distribution\n\n")
        f.write(f"{sites.shape[0]} sites x {sites.shape[1]} species. Full table: `per_site_distribution.csv`.\n")
        sites.to_csv(REPORT_PATH.parent / "per_site_distribution.csv")

    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
