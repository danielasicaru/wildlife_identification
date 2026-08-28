# Camera Trap Species Classifier

Automated wildlife species identification from camera trap imagery, evaluated across multiple
camera sites to expose cross-location generalization gaps rather than optimizing for a single
deployment environment.

## Problem

Camera traps deployed across national parks generate large volumes of images that require manual
review to identify species. A model that classifies reliably as it moves between camera sites with
different backgrounds, lighting, and framing reduces that review burden without overfitting to one
location's conditions.

## Approach

A five-stage pipeline: dataset characterization, augmentation strategy, animal localization,
classifier training (controlled multi-backbone comparison), and evaluation, ending in a
production-shaped FastAPI inference service.

**Dataset:** Caltech Camera Traps (via LILA BC), paired with the "Recognition in Terra Incognita"
benchmark for cross-site generalization.

**Stack:** Python, PyTorch, MegaDetector, FastAPI, MLflow, Docker.

**Status:** dataset characterization, augmentation, localization, classifier training, and
evaluation complete. Serving not yet started.

## Setup

Exact reproduction of the environment this was built in:

```bash
conda env create -f environment.yml
conda activate wildlife-id
```

Or manually:

```bash
conda create -n wildlife-id python=3.11
conda activate wildlife-id
pip install pandas pytest matplotlib jupyter nbconvert ipykernel opencv-python-headless imagehash numpy pillow pyyaml tabulate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # or the CPU index
pip install --no-deps megadetector clipboard dill fastquadtree humanfriendly jsonpickle mkl \
    scikit-learn thop ultralytics-yolov5 seaborn  # --no-deps: avoids a conflict with opencv-python-headless
pip install mlflow
```

Pipeline, in order:

```bash
python scripts/download_sample_images.py      # requires caltech_images_20210113.json in data/raw/
python scripts/run_localization.py            # downloads MegaDetector v5a weights on first run
python scripts/generate_localization_report.py
python scripts/generate_data_manifest.py       # optional, referenced by train_classifier.py if present
python scripts/train_classifier.py             # MLflow tracking, local file store
python scripts/evaluate_classifier.py
python scripts/evaluate_detector.py
python -m pytest tests/ -v
```

## Dataset characterization

Structure inspection, class balance, image statistics, and quality checks against the real
243,100-image annotation set and a 633-image stratified sample.

- [notebooks/eda.ipynb](notebooks/eda.ipynb) — visual walkthrough with rendered charts
- `reports/characterization.md`, `reports/quality.md`, `reports/gap_analysis.md` — regenerate via
  the matching `scripts/generate_*_report.py`

### Key findings

- **Severe class imbalance**: 8349x between the most and least common species (opossum vs. pig) —
  drives the oversampling/differential-augmentation design below.
- **The hour-of-day day/night proxy is unreliable**: pixel-based grayscale detection found 59.9%
  of the sample effectively grayscale vs. the metadata proxy's 40% — a 19.9-point gap. Downstream
  code uses the pixel-based check directly, not the proxy.
- **61 near-duplicate pairs** found via perceptual hashing — kept together across train/val/test.
- **Zero corrupted images** in the sample; one malformed `date_captured` handled without crashing.
- **Extreme scale variation**: median animal occupies under 3% of frame area, a third under 2% —
  drove the augmentation crop-scale range being widened from 0.7-1.0 to 0.4-1.0.
- **Occlusion has no automatic proxy**; a 20-image manual sample: 40% heavy, 35% clear, 25% partial.

## Augmentation pipeline

`src/data/augmentation.py`: `torchvision.transforms.v2` pipelines with differential per-class
probabilities and a `WeightedRandomSampler`-ready weighting function. 11 unit tests.

- [notebooks/eda.ipynb](notebooks/eda.ipynb) — before/after grids, majority-vs-minority
  comparison, oversampling effect chart

### Deviations from the original design

- **Crop scale 0.4-1.0**, not 0.7-1.0 — the gap analysis above found the narrower range couldn't
  represent the small-animal tail.
- **Gaussian noise sigma is a fixed 0.02**, not a sampled range — `torchvision`'s `GaussianNoise`
  has no native range support.
- **Normalization uses ImageNet mean/std**, matching the pretrained-backbone approach.

Wired into a `Dataset`/`DataLoader` in `src/classifier/dataset.py`.

## Localization

`src/localization/` wraps MegaDetector v5a as an inference-only detector: model loading/inference/
filtering (`detector.py`), bbox expansion and cropping (`crop.py`), IoU-based recall (`evaluate.py`).
13 unit tests plus one integration test against the real model.

- `scripts/run_localization.py` — detection + cropping over the sample
- `scripts/generate_localization_report.py` — recall vs. ground truth (`reports/localization.md`)
- [notebooks/eda.ipynb](notebooks/eda.ipynb) — bounding boxes and resulting crops

### Key findings

- **531 of 633 images** produced at least one detection (617 crops) at the documented "typical"
  confidence threshold (0.2).
- **92.5% recall** (331/358 ground-truth animals, IoU >= 0.5) — a small-sample sanity check, not
  full detector evaluation.
- MegaDetector v5a chosen over the newer "v1000" family for better-documented behavior.

## Classifier training

`src/classifier/`: per-crop species labeling via ground-truth IoU matching with a single-species
fallback (`labeling.py`), near-duplicate-aware train/val/test split (`split.py`), a `Dataset`
reusing the augmentation pipeline (`dataset.py`), a backbone factory for ResNet50/EfficientNet-B0/
ViT-B/16 (`models.py`), and class-weighted train/evaluate loops plus early stopping (`engine.py`).
26 unit tests.

- `scripts/train_classifier.py` — trains + compares all three backbones with class-weighted loss,
  a `WeightedRandomSampler`, and early stopping (patience in `configs/train_classifier.yaml`,
  restores best-epoch weights on stop); logs params/metrics/checkpoints to local-file MLflow
- [notebooks/eda.ipynb](notebooks/eda.ipynb) — per-backbone metrics and comparison chart

### Key findings

- **594 crops labeled across 19 species** from 617 crops (a handful dropped as genuinely ambiguous
  multi-species images). Split 416/90/88 (train/val/test).
- **Early stopping made a real difference**: ResNet50 stopped at epoch 19 (best epoch 9, 38.9% val
  accuracy), EfficientNet-B0 at 37 (best epoch 27, 51.1%), ViT-B/16 at 15 (best epoch 5, 48.9%) —
  up from a flat 5-epoch budget. Best-epoch weights (by `val_loss`) are restored and checkpointed
  on stop, not whatever the final epoch produced. This matters in practice: ViT-B/16's *accuracy*
  kept climbing past its best-*loss* epoch (57.8% by epoch 15 vs. 48.9% at epoch 5) — loss and
  accuracy don't always move together on a 90-example validation set, so which one you select on
  changes the checkpoint you end up with. This is a single seed-locked run on ~500 crops across
  imbalanced classes, read as a pipeline-correctness check, not a performance benchmark.

## Evaluation

`src/evaluation/`: per-class metrics via `scikit-learn` (`classifier_metrics.py`), pixel-based
day/night and per-site segmentation (`segmentation.py`), and a from-scratch single-class detector
AP (`detector_metrics.py`). Trained checkpoints and species-index mappings are logged as MLflow
artifacts so a model can be reloaded independently of its training run.

- `scripts/evaluate_classifier.py` — best backbone, held-out test split: per-class P/R/F1,
  confusion matrix, day/night and per-site breakdowns, full misclassification list
- `scripts/evaluate_detector.py` — average precision, missed-detection analysis by size/day-night
- [notebooks/eda.ipynb](notebooks/eda.ipynb) — confusion matrix and raw report text

### Key findings

- **Classifier: 47.7% test accuracy** (EfficientNet-B0, the best-val-loss backbone once
  best-epoch-weight restoration was fixed) on 88 test crops across 19 species. Per-class numbers
  vary widely by support, as expected at this scale. Day/night accuracy gap is now small (48.3% vs.
  47.5%) — down from a 25-point gap before early stopping existed at all.
- **Detector: 0.535 Average Precision** (IoU >= 0.5) — lower than the 92.5% raw recall since AP
  also penalizes false positives. Counter-intuitively, large animals (>10% of frame) had the
  lowest per-box detection rate (81.6%) vs. small (88.2%) and medium (100%).
- **Occlusion segmentation is intentionally skipped**: only 20 manually tagged images against an
  88-crop test set — not enough overlap for a real finding.

## Reproducibility

- **Environment**: `environment.yml` (`conda env export`) — most packages here were pip-installed,
  so this is the accurate record, not a hand-written requirements list.
- **Seeds**: `random`/`numpy`/`torch` locked, `cudnn.deterministic`/`benchmark` set, an explicit
  seeded `torch.Generator` passed to `WeightedRandomSampler`.
- **Config, not inline settings**: per-script settings live in `configs/*.yaml` via
  `src/utils/config.py`.
- **Data versioning**: `scripts/generate_data_manifest.py` writes `reports/data_manifest.json`
  (SHA-256 per file/directory) since `data/` itself is gitignored.
- **Automatic experiment metadata**: every training run logs its config, data manifest, and
  Python/PyTorch/CUDA versions to MLflow alongside the metrics.
- A checksum manifest was chosen over DVC, and plain YAML over Hydra, to match this project's
  scale.

## Tradeoffs

- Only annotation metadata was downloaded initially, not the full ~105GB archive; a 633-image
  stratified sample was pulled once real pixel data was needed, trading full-dataset exhaustiveness
  for fast iteration.
- MegaDetector (pretrained) is used for localization instead of training a custom detector, to
  keep effort on classification/deployment.
- The classifier split isn't hard-stratified by class — per-species counts as low as single digits
  make that infeasible without dropping already-scarce species. Some rare species end up
  train-only as a documented consequence of the imbalance found during characterization.

## Repository conventions

- Prefer small, focused files over large ones that do too much.
- Follow existing patterns before introducing new ones.
- Commit messages, PR descriptions, code comments, and README content are written in first person,
  as the repository author.
