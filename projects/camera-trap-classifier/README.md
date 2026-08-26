# Camera Trap Species Classifier

Automated wildlife species identification from camera trap imagery, evaluated across multiple
camera sites to expose cross-location generalization gaps rather than optimizing for a single
deployment environment.

## Problem

Camera traps deployed across national parks generate large volumes of images that require manual
review to identify species. A model that classifies species — and does so reliably as it moves
between camera sites with different backgrounds, lighting, and framing — reduces that review
burden and generalizes to new deployments rather than overfitting to one location's conditions.

## Approach

A five-stage pipeline: dataset characterization, augmentation strategy, animal localization,
classifier training (with a controlled multi-backbone comparison), and evaluation, ending in a
production-shaped FastAPI inference service.

**Dataset:** Caltech Camera Traps (via LILA BC), paired with the "Recognition in Terra Incognita"
benchmark for cross-site generalization.

**Status:** dataset characterization and augmentation pipeline functionally complete. One manual
step remains open: occlusion tagging (see "Key findings" below). Localization, classifier
training, evaluation, and serving not yet started.

## Setup

```bash
conda create -n wildlife-id python=3.11
conda activate wildlife-id
pip install pandas pytest matplotlib jupyter nbconvert ipykernel opencv-python-headless imagehash numpy pillow
```

Download the annotation files and a stratified image sample:

```bash
python scripts/download_sample_images.py  # requires caltech_images_20210113.json in data/raw/ first
```

Run tests:

```bash
python -m pytest tests/ -v
```

## Dataset characterization

Structure inspection, class balance, image statistics, and quality checks, run against the real
243,100-image annotation set and a 390-image stratified sample (~20 images per species).

- [notebooks/eda.ipynb](notebooks/eda.ipynb) — visual walkthrough with rendered charts
- `reports/characterization.md`, `reports/quality.md`, `reports/gap_analysis.md` — generated
  reports (not committed; run `scripts/generate_characterization_report.py`,
  `scripts/generate_quality_report.py`, `scripts/generate_gap_analysis_report.py`)

### Key findings

- **Severe class imbalance**: 8349x between the most and least common species (opossum vs. pig).
  Directly informs the oversampling and differential-augmentation-probability design in the
  augmentation spec.
- **The hour-of-day day/night proxy is unreliable**: pixel-based grayscale detection found 57.4%
  of the sample effectively grayscale, a 17-point gap from the metadata proxy's 40% night
  estimate. Decision: use the pixel-based check directly wherever day/night status affects a
  downstream choice (e.g. the augmentation pipeline's selective grayscale targeting), rather than
  the metadata proxy.
- **26 near-duplicate pairs** found in the 390-image sample via perceptual hashing — a real signal
  to deduplicate before creating train/val/test splits.
- **Zero corrupted images** in the sample; not a concern at current scale, worth a full-dataset
  pass before final training.
- One malformed `date_captured` value out of 243,100 — negligible in volume, handled without
  crashing rather than ignored.
- **Extreme scale variation**: from bounding-box annotations (65,112 boxes, 63,025 images — a
  partial subset), the median animal occupies under 3% of frame area, and a third of annotated
  images are under 2%. More extreme than a casual "animals appear at different distances"
  assumption. Flagged as a follow-up to revisit against the augmentation spec's crop-scale range
  when augmentation is implemented.
- **Occlusion has no reliable automatic proxy** without segmentation masks. `notebooks/eda.ipynb`
  has a manual tagging tool — run it yourself, tag a sample grid as clear/partial/heavy, save to
  `data/occlusion_tags.json`, then re-run `scripts/generate_gap_analysis_report.py`. This step is
  intentionally left open rather than faked.

## Augmentation pipeline

`src/data/augmentation.py` implements the design spec's training/validation transform pipelines
with `torchvision.transforms.v2`, differential per-class probabilities, and a
`WeightedRandomSampler`-ready weighting function. 11 unit tests cover shape/dtype correctness,
crop-wrapping behavior (minority applied unconditionally, majority wrapped at p=0.8), and
per-step probability differences between majority and minority classes.

- [notebooks/eda.ipynb](notebooks/eda.ipynb) — before/after augmented grids (majority pipeline,
  and a majority-vs-minority side-by-side with a fixed seed), plus a raw-count-vs-expected-draws
  chart showing the oversampling mechanism's effect on effective class balance

### Deviations from the original spec

- **Crop scale widened from 0.7-1.0 to 0.4-1.0** after the gap analysis found a third of annotated
  animals occupy under 2% of frame area — the original range couldn't reach far enough to
  represent that tail during training.
- **Gaussian noise sigma is a fixed 0.02**, not a sampled 0.01-0.03 range — `torchvision`'s
  `GaussianNoise` doesn't support range sampling natively, so the pipeline uses the midpoint of
  the originally planned range instead.
- **Normalization uses ImageNet mean/std**, not dataset-computed statistics, matching the
  pretrained-backbone transfer-learning approach planned for classifier training.

### Not yet done

The pipeline exists as standalone `build_train_transform`/`build_val_transform` functions; it
isn't wired into a `Dataset`/`DataLoader` yet, since that depends on the localization stage's
cropped-image outputs, which don't exist until MegaDetector integration is implemented.

## Tradeoffs

- Only annotation metadata was downloaded initially, not the full ~105GB image archive; a small
  stratified sample (390 images) was pulled once image statistics and quality checks required
  real pixel data. This kept iteration fast without blocking on a multi-hour download, at the cost
  of quality-check findings being sample-based rather than exhaustive until a full-dataset pass.
- MegaDetector (pretrained) is used for localization rather than training a custom detector, to
  keep engineering effort focused on classification, optimization, and deployment rather than
  object detection research.
