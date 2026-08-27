"""Builds notebooks/eda.ipynb as source (no outputs yet -- run nbconvert --execute after)."""
from pathlib import Path

import nbformat as nbf

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "eda.ipynb"

nb = nbf.v4.new_notebook()
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        "# Caltech Camera Traps — Exploratory Data Analysis\n\n"
        "Visual companion to `reports/characterization.md`. Uses the same "
        "`src.data.characterize` functions the tests cover, so the numbers here match the "
        "report exactly."
    ),
    nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n\n"
        "sys.path.insert(0, str(Path.cwd().parent))\n\n"
        "from src.data.loader import load_annotations\n"
        "from src.data.characterize import class_distribution, metadata_survey, per_site_distribution\n\n"
        "NON_SPECIES = {\"empty\", \"car\"}\n"
        "ANNOTATIONS_PATH = Path.cwd().parent / \"data\" / \"raw\" / \"caltech_images_20210113.json\"\n\n"
        "images, annotations, categories = load_annotations(ANNOTATIONS_PATH)\n"
        "print(f\"{len(images)} images, {len(annotations)} annotations, {images['location'].nunique()} sites\")"
    ),
    nbf.v4.new_markdown_cell("## Class distribution\n\nSpecies counts, excluding non-species categories (`empty`, `car`)."),
    nbf.v4.new_code_cell(
        "classes = class_distribution(annotations, categories, exclude=NON_SPECIES)\n\n"
        "fig, ax = plt.subplots(figsize=(10, 6))\n"
        "classes[\"count\"].plot.barh(ax=ax)\n"
        "ax.set_xlabel(\"Annotation count\")\n"
        "ax.set_title(\"Species distribution (Caltech Camera Traps)\")\n"
        "ax.invert_yaxis()\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "print(f\"Imbalance ratio (majority/minority): {classes['imbalance_ratio'].max():.1f}x\")"
    ),
    nbf.v4.new_markdown_cell(
        "## Per-site distribution\n\n"
        "Species counts by camera location — exposes the cross-site domain gap models need to "
        "generalize across. Showing the 15 highest-traffic sites for readability; the full "
        "140-site table is in "
        "`reports/per_site_distribution.csv`."
    ),
    nbf.v4.new_code_cell(
        "sites = per_site_distribution(images, annotations, categories, exclude=NON_SPECIES)\n"
        "top_sites = sites.loc[sites.sum(axis=1).sort_values(ascending=False).head(15).index]\n\n"
        "fig, ax = plt.subplots(figsize=(12, 7))\n"
        "im = ax.imshow(top_sites.values, aspect=\"auto\", cmap=\"viridis\")\n"
        "ax.set_xticks(range(len(top_sites.columns)))\n"
        "ax.set_xticklabels(top_sites.columns, rotation=90)\n"
        "ax.set_yticks(range(len(top_sites.index)))\n"
        "ax.set_yticklabels(top_sites.index)\n"
        "ax.set_xlabel(\"Species\")\n"
        "ax.set_ylabel(\"Site (location id)\")\n"
        "ax.set_title(\"Species counts, top 15 sites by traffic\")\n"
        "fig.colorbar(im, ax=ax, label=\"Count\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell("## Metadata survey\n\nResolution mix and the day/night hour-of-day proxy (see docstring in `characterize.py` for why this is a proxy, not ground truth)."),
    nbf.v4.new_code_cell(
        "meta = metadata_survey(images)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
        "resolutions = {f\"{w}x{h}\": count for (h, w), count in meta[\"resolution_counts\"].items()}\n"
        "axes[0].bar(resolutions.keys(), resolutions.values())\n"
        "axes[0].set_title(\"Resolution counts\")\n"
        "axes[0].tick_params(axis=\"x\", rotation=45)\n\n"
        "axes[1].pie(\n"
        "    [meta[\"day_count\"], meta[\"night_count\"], meta[\"unparseable_date_count\"]],\n"
        "    labels=[\"Day (proxy)\", \"Night (proxy)\", \"Unparseable date\"],\n"
        "    autopct=\"%1.1f%%\",\n"
        ")\n"
        "axes[1].set_title(\"Day/night proxy split\")\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell("## Aspect ratio\n\nBucketed from the same width/height metadata used for resolution counts."),
    nbf.v4.new_code_cell(
        "from src.data.characterize import aspect_ratio_survey\n\n"
        "aspect = aspect_ratio_survey(images)\n"
        "fig, ax = plt.subplots(figsize=(5, 4))\n"
        "ax.bar(aspect.keys(), aspect.values())\n"
        "ax.set_title(\"Aspect ratio buckets\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## Image quality checks (sample)\n\n"
        "Run on a stratified sample of real downloaded images (`data/raw/images/`, ~20 per "
        "category, not the full dataset) — annotations alone can't tell us about corruption, "
        "blur, or actual color content. See `reports/quality.md` for the full findings-and-"
        "recommendations writeup; this section is the visual companion."
    ),
    nbf.v4.new_code_cell(
        "from src.data.quality import blur_score, find_near_duplicates, is_corrupted, is_effectively_grayscale\n\n"
        "IMAGES_DIR = Path.cwd().parent / \"data\" / \"raw\" / \"images\"\n"
        "sample_paths = sorted(IMAGES_DIR.glob(\"*.jpg\"))\n"
        "corrupted = [p for p in sample_paths if is_corrupted(p)]\n"
        "valid_paths = [p for p in sample_paths if p not in corrupted]\n"
        "scores = {p: blur_score(p) for p in valid_paths}\n"
        "print(f\"{len(sample_paths)} sample images, {len(corrupted)} corrupted\")"
    ),
    nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "ax.hist(list(scores.values()), bins=30)\n"
        "ax.set_xlabel(\"Laplacian variance (higher = sharper)\")\n"
        "ax.set_ylabel(\"Count\")\n"
        "ax.set_title(\"Blur score distribution\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell("Sharpest vs. blurriest images in the sample, side by side:"),
    nbf.v4.new_code_cell(
        "from PIL import Image as PILImage\n\n"
        "sharpest = max(scores, key=scores.get)\n"
        "blurriest = min(scores, key=scores.get)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n"
        "axes[0].imshow(PILImage.open(sharpest))\n"
        "axes[0].set_title(f\"Sharpest (score={scores[sharpest]:.0f})\")\n"
        "axes[0].axis(\"off\")\n"
        "axes[1].imshow(PILImage.open(blurriest))\n"
        "axes[1].set_title(f\"Blurriest (score={scores[blurriest]:.0f})\")\n"
        "axes[1].axis(\"off\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "Pixel-based grayscale detection vs. the metadata hour-of-day proxy — see "
        "`reports/quality.md` for why these disagree and what to do about it:"
    ),
    nbf.v4.new_code_cell(
        "n_gray = sum(is_effectively_grayscale(p) for p in valid_paths)\n"
        "pixel_gray_share = n_gray / len(valid_paths)\n"
        "hour_proxy_night_share = meta[\"night_count\"] / (meta[\"day_count\"] + meta[\"night_count\"])\n\n"
        "fig, ax = plt.subplots(figsize=(5, 4))\n"
        "ax.bar(\n"
        "    [\"Pixel-based\\n(grayscale check)\", \"Hour-based\\n(proxy)\"],\n"
        "    [pixel_gray_share, hour_proxy_night_share],\n"
        ")\n"
        "ax.set_ylabel(\"Share flagged as night/grayscale\")\n"
        "ax.set_title(\"Day/night estimate: pixel evidence vs. metadata proxy\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell("A near-duplicate pair caught by perceptual hashing (if any exist in this sample):"),
    nbf.v4.new_code_cell(
        "duplicates = find_near_duplicates(valid_paths)\n"
        "print(f\"{len(duplicates)} near-duplicate pairs found\")\n\n"
        "if duplicates:\n"
        "    path_a, path_b = duplicates[0]\n"
        "    fig, axes = plt.subplots(1, 2, figsize=(8, 4))\n"
        "    axes[0].imshow(PILImage.open(path_a))\n"
        "    axes[0].set_title(path_a.name)\n"
        "    axes[0].axis(\"off\")\n"
        "    axes[1].imshow(PILImage.open(path_b))\n"
        "    axes[1].set_title(path_b.name)\n"
        "    axes[1].axis(\"off\")\n"
        "    plt.tight_layout()\n"
        "    plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## Scale variation\n\n"
        "Bounding-box area as a fraction of image area — how much of the frame the animal "
        "actually occupies, from `data/raw/caltech_bboxes_20200316.json` (a 63,025-image subset "
        "with bbox annotations, not the full dataset). See `reports/gap_analysis.md` for the "
        "full writeup."
    ),
    nbf.v4.new_code_cell(
        "import json\n"
        "from src.data.characterize import bbox_area_ratio\n\n"
        "BBOX_PATH = Path.cwd().parent / \"data\" / \"raw\" / \"caltech_bboxes_20200316.json\"\n"
        "with open(BBOX_PATH) as bf:\n"
        "    bbox_data = json.load(bf)\n"
        "bboxes = pd.DataFrame(bbox_data[\"annotations\"])\n"
        "bbox_images = pd.DataFrame(bbox_data[\"images\"])\n\n"
        "ratios = bbox_area_ratio(bboxes, bbox_images)[\"area_ratio\"]\n"
        "print(f\"median area ratio: {ratios.median():.3f}, \"\n"
        "      f\"under 2% of frame: {100*(ratios < 0.02).mean():.1f}%\")"
    ),
    nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "ax.hist(ratios, bins=50, range=(0, 0.3))\n"
        "ax.set_xlabel(\"Bounding box area / image area\")\n"
        "ax.set_ylabel(\"Count\")\n"
        "ax.set_title(\"Scale variation (clipped at 0.3 for readability; long tail continues to 1.0)\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## Occlusion — manual tagging tool\n\n"
        "No reliable automatic proxy exists for occlusion without segmentation masks, so this is "
        "a deliberately manual, hands-on step (see the pair-programming decision behind this "
        "choice). **Run this section yourself in Jupyter**, not just read it:\n\n"
        "1. Run the grid cell below to view a stratified batch of sample images.\n"
        "2. For each filename shown, decide: `clear` (animal fully visible), `partial` (some "
        "occlusion — a branch, grass, partial frame cutoff), or `heavy` (animal mostly hidden).\n"
        "3. Fill in the `occlusion_tags` dict in the cell after the grid with your judgments.\n"
        "4. Run the save cell to persist your tags to `data/occlusion_tags.json`.\n"
        "5. Re-run `python scripts/generate_gap_analysis_report.py` to fold your tags into the "
        "report."
    ),
    nbf.v4.new_code_cell(
        "import random\n\n"
        "random.seed(42)\n"
        "tagging_batch = random.sample(valid_paths, min(20, len(valid_paths)))\n\n"
        "fig, axes = plt.subplots(4, 5, figsize=(18, 15))\n"
        "for ax, path in zip(axes.flat, tagging_batch):\n"
        "    ax.imshow(PILImage.open(path))\n"
        "    ax.set_title(path.name, fontsize=8)\n"
        "    ax.axis(\"off\")\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "print(\"Filenames to tag:\")\n"
        "for path in tagging_batch:\n"
        "    print(f\"  {path.name!r}: \\\"\\\",  # clear / partial / heavy\")"
    ),
    nbf.v4.new_code_cell(
        "# Fill this in by hand after looking at the grid above -- replace with your own judgments.\n"
        "# This is intentionally left empty here; running it as-is produces no tags, which is\n"
        "# correct until you've actually done the visual inspection.\n"
        "occlusion_tags = {\n"
        "    # \"5968c0f9-23d2-11e8-a6a3-ec086b02610b.jpg\": \"clear\",\n"
        "    # \"5998cfa4-23d2-11e8-a6a3-ec086b02610b.jpg\": \"partial\",\n"
        "}\n\n"
        "if occlusion_tags:\n"
        "    print(pd.Series(occlusion_tags).value_counts())\n"
        "else:\n"
        "    print(\"No tags recorded yet -- fill in occlusion_tags above after reviewing the grid.\")"
    ),
    nbf.v4.new_code_cell(
        "OCCLUSION_TAGS_PATH = Path.cwd().parent / \"data\" / \"occlusion_tags.json\"\n\n"
        "if occlusion_tags:\n"
        "    with open(OCCLUSION_TAGS_PATH, \"w\") as tf:\n"
        "        json.dump(occlusion_tags, tf, indent=2)\n"
        "    print(f\"Saved {len(occlusion_tags)} tags to {OCCLUSION_TAGS_PATH}\")\n"
        "else:\n"
        "    print(\"Nothing to save -- occlusion_tags is empty.\")"
    ),
    nbf.v4.new_markdown_cell(
        "## Augmentation outputs\n\n"
        "Before/after examples using the actual training transform pipeline "
        "(`src.data.augmentation.build_train_transform`)."
    ),
    nbf.v4.new_code_cell(
        "import torch\n"
        "from src.data.augmentation import build_train_transform, minority_species\n\n"
        "aug_minority = minority_species(classes[\"count\"])\n"
        "print(f\"{len(aug_minority)} of {len(classes)} species are minority class\")"
    ),
    nbf.v4.new_code_cell(
        "IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)\n"
        "IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)\n\n"
        "def denormalize(tensor):\n"
        "    return (tensor * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)\n\n"
        "aug_sample_path = valid_paths[0]\n"
        "aug_original = PILImage.open(aug_sample_path)\n\n"
        "aug_train_transform = build_train_transform(is_minority=False)\n\n"
        "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n"
        "axes[0, 0].imshow(aug_original)\n"
        "axes[0, 0].set_title(\"Original\")\n"
        "axes[0, 0].axis(\"off\")\n\n"
        "for i in range(1, 8):\n"
        "    ax = axes.flat[i]\n"
        "    augmented = aug_train_transform(aug_original)\n"
        "    ax.imshow(denormalize(augmented).permute(1, 2, 0).numpy())\n"
        "    ax.set_title(f\"Augmented {i}\")\n"
        "    ax.axis(\"off\")\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "### Majority vs. minority: differential augmentation probability\n\n"
        "Same source image run through both pipeline variants, holding the random seed fixed so "
        "any visual difference comes from the per-class probability/parameter differences, not "
        "from random chance."
    ),
    nbf.v4.new_code_cell(
        "aug_train_transform_minority = build_train_transform(is_minority=True)\n\n"
        "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n"
        "for row, (label, transform) in enumerate(\n"
        "    [(\"Majority\", aug_train_transform), (\"Minority\", aug_train_transform_minority)]\n"
        "):\n"
        "    torch.manual_seed(0)\n"
        "    for col in range(4):\n"
        "        ax = axes[row, col]\n"
        "        augmented = transform(aug_original)\n"
        "        ax.imshow(denormalize(augmented).permute(1, 2, 0).numpy())\n"
        "        ax.set_title(f\"{label} {col + 1}\")\n"
        "        ax.axis(\"off\")\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "### Oversampling effect on effective class balance\n\n"
        "Raw annotation counts vs. the expected number of times each species is drawn per epoch "
        "under a `WeightedRandomSampler` built from `build_sample_weights` — the mechanism that "
        "actually balances effective training-set size, distinct from the per-step augmentation "
        "probabilities above."
    ),
    nbf.v4.new_code_cell(
        "from src.data.augmentation import build_sample_weights\n\n"
        "counts = classes[\"count\"]\n"
        "labels = pd.Series(counts.index.repeat(counts.values))\n"
        "weights = pd.Series(build_sample_weights(labels, counts), index=labels.index)\n\n"
        "total_weight = weights.sum()\n"
        "total_samples = len(labels)\n"
        "expected_draws = (\n"
        "    pd.DataFrame({\"label\": labels, \"weight\": weights})\n"
        "    .groupby(\"label\")[\"weight\"]\n"
        "    .sum()\n"
        "    .div(total_weight)\n"
        "    .mul(total_samples)\n"
        ")\n\n"
        "comparison = pd.DataFrame({\"raw_count\": counts, \"expected_draws_per_epoch\": expected_draws})\n"
        "comparison = comparison.sort_values(\"raw_count\")\n\n"
        "fig, ax = plt.subplots(figsize=(10, 6))\n"
        "x = range(len(comparison))\n"
        "ax.barh(x, comparison[\"raw_count\"], height=0.4, label=\"Raw count\", align=\"edge\")\n"
        "ax.barh([i + 0.4 for i in x], comparison[\"expected_draws_per_epoch\"], height=0.4,\n"
        "        label=\"Expected draws/epoch (weighted sampler)\", align=\"edge\")\n"
        "ax.set_yticks([i + 0.4 for i in x])\n"
        "ax.set_yticklabels(comparison.index)\n"
        "ax.set_xscale(\"log\")\n"
        "ax.set_xlabel(\"Count (log scale)\")\n"
        "ax.set_title(\"Raw class imbalance vs. oversampling-corrected effective balance\")\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## Localization (MegaDetector)\n\n"
        "Bounding boxes and crops produced by `scripts/run_localization.py` "
        "(`src/localization/detector.py`, `src/localization/crop.py`)."
    ),
    nbf.v4.new_code_cell(
        "import json\n"
        "from PIL import Image as PILImage, ImageDraw\n\n"
        "LOCALIZATION_DIR = Path.cwd().parent / \"data\" / \"localization\"\n"
        "with open(LOCALIZATION_DIR / \"detections.json\") as lf:\n"
        "    localization_results = json.load(lf)\n\n"
        "with_animal = [r for r in localization_results if r[\"crops\"]]\n"
        "print(f\"{len(with_animal)}/{len(localization_results)} sample images had an animal detection\")"
    ),
    nbf.v4.new_markdown_cell("Detected bounding boxes drawn on the original frame, for a few sample images:"),
    nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n"
        "for ax, result in zip(axes.flat, with_animal[:6]):\n"
        "    img_path = IMAGES_DIR / result[\"source_image\"]\n"
        "    img = PILImage.open(img_path).convert(\"RGB\")\n"
        "    draw = ImageDraw.Draw(img)\n"
        "    for crop_info in result[\"crops\"]:\n"
        "        x, y, w, h = crop_info[\"bbox_absolute\"]\n"
        "        draw.rectangle([x, y, x + w, y + h], outline=\"red\", width=6)\n"
        "    ax.imshow(img)\n"
        "    ax.set_title(f\"{result['source_image']}\", fontsize=8)\n"
        "    ax.axis(\"off\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell("Cropped animal images that will feed the classifier:"),
    nbf.v4.new_code_cell(
        "crop_paths = sorted((LOCALIZATION_DIR / \"crops\").glob(\"*.jpg\"))[:8]\n"
        "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n"
        "for ax, crop_path in zip(axes.flat, crop_paths):\n"
        "    ax.imshow(PILImage.open(crop_path))\n"
        "    ax.set_title(crop_path.name, fontsize=7)\n"
        "    ax.axis(\"off\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## Classifier training comparison\n\n"
        "Per-epoch metrics logged by `scripts/train_classifier.py` to local-file MLflow "
        "(`mlruns/`, gitignored). Small-sample sanity run on MegaDetector crops, not a "
        "final-performance claim -- no full evaluation (per-class precision/recall, confusion "
        "matrix, failure analysis) has been run yet."
    ),
    nbf.v4.new_code_cell(
        "import os\n"
        "os.environ.setdefault(\"MLFLOW_ALLOW_FILE_STORE\", \"true\")\n"
        "import mlflow\n\n"
        "mlflow.set_tracking_uri((Path.cwd().parent / \"mlruns\").as_uri())\n"
        "experiment = mlflow.get_experiment_by_name(\"camera-trap-classifier\")\n"
        "runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id]) if experiment else None\n"
        "if runs is not None and not runs.empty:\n"
        "    display_cols = [c for c in runs.columns if c.startswith(\"params.\") or c.startswith(\"metrics.final\")]\n"
        "    print(runs[[\"tags.mlflow.runName\"] + display_cols].to_string(index=False))\n"
        "else:\n"
        "    print(\"No MLflow runs found -- run scripts/train_classifier.py first.\")"
    ),
    nbf.v4.new_markdown_cell("Validation accuracy per backbone, final epoch:"),
    nbf.v4.new_code_cell(
        "if runs is not None and not runs.empty:\n"
        "    fig, ax = plt.subplots(figsize=(6, 4))\n"
        "    ax.bar(runs[\"tags.mlflow.runName\"], runs[\"metrics.final_val_accuracy\"])\n"
        "    ax.set_ylabel(\"Final validation accuracy\")\n"
        "    ax.set_title(\"Backbone comparison (small-sample sanity run)\")\n"
        "    plt.tight_layout()\n"
        "    plt.show()"
    ),
]

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written to {NOTEBOOK_PATH}")
