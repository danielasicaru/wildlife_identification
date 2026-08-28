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

**Stack:** Python, PyTorch, MegaDetector, FastAPI, MLflow, Docker.

**Status:** dataset characterization, augmentation pipeline, localization, classifier training, and
evaluation functionally complete. Serving not yet started.

## Setup

For an exact reproduction of the environment this was built and run in:

```bash
conda env create -f environment.yml
conda activate wildlife-id
```

Or set it up manually:

```bash
conda create -n wildlife-id python=3.11
conda activate wildlife-id
pip install pandas pytest matplotlib jupyter nbconvert ipykernel opencv-python-headless imagehash numpy pillow pyyaml tabulate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # or the CPU index if no GPU
pip install --no-deps megadetector clipboard dill fastquadtree humanfriendly jsonpickle mkl \
    scikit-learn thop ultralytics-yolov5 seaborn  # --no-deps avoids a pip conflict between
    # megadetector's opencv-python requirement and the already-installed opencv-python-headless
pip install mlflow  # full install is fine here -- no opencv dependency to conflict with
```

Download the annotation files and a stratified image sample:

```bash
python scripts/download_sample_images.py  # requires caltech_images_20210113.json in data/raw/ first
```

Run localization (downloads the ~280MB MegaDetector v5a weights on first run):

```bash
python scripts/run_localization.py
python scripts/generate_localization_report.py
```

Generate a checksummed manifest of the exact data files in use (optional, but referenced by
`train_classifier.py` if present):

```bash
python scripts/generate_data_manifest.py
```

Train and compare classifier backbones (MLflow tracking, local file store):

```bash
python scripts/train_classifier.py
```

Evaluate the best trained model on the held-out test split, and the detector's average precision:

```bash
python scripts/evaluate_classifier.py
python scripts/evaluate_detector.py
```

Run tests:

```bash
python -m pytest tests/ -v
```

## Dataset characterization

Structure inspection, class balance, image statistics, and quality checks, run against the real
243,100-image annotation set and a 633-image stratified sample (~20 images per category,
including `empty`).

- [notebooks/eda.ipynb](notebooks/eda.ipynb) — visual walkthrough with rendered charts
- `reports/characterization.md`, `reports/quality.md`, `reports/gap_analysis.md` — generated
  reports (run `scripts/generate_characterization_report.py`, `scripts/generate_quality_report.py`,
  `scripts/generate_gap_analysis_report.py` to regenerate)

### Key findings

- **Severe class imbalance**: 8349x between the most and least common species (opossum vs. pig).
  Directly informs the oversampling and differential-augmentation-probability design in the
  augmentation pipeline below.
- **The hour-of-day day/night proxy is unreliable**: pixel-based grayscale detection found 59.9%
  of the sample effectively grayscale, a 19.9-point gap from the metadata proxy's 40% night
  estimate. Decision: use the pixel-based check directly wherever day/night status affects a
  downstream choice (e.g. the augmentation pipeline's selective grayscale targeting), rather than
  the metadata proxy.
- **61 near-duplicate pairs** found in the 633-image sample via perceptual hashing — a real signal
  to deduplicate before creating train/val/test splits.
- **Zero corrupted images** in the sample; not a concern at current scale, worth a full-dataset
  pass before final training.
- One malformed `date_captured` value out of 243,100 — negligible in volume, handled without
  crashing rather than ignored.
- **Extreme scale variation**: from bounding-box annotations (65,112 boxes, 63,025 images — a
  partial subset), the median animal occupies under 3% of frame area, and a third of annotated
  images are under 2%. More extreme than a casual "animals appear at different distances"
  assumption. This is what drove the augmentation pipeline's crop-scale range being widened from
  0.7-1.0 to 0.4-1.0 (see the augmentation section below).
- **Occlusion has no reliable automatic proxy** without segmentation masks, so a 20-image sample
  was manually tagged clear/partial/heavy via the `notebooks/eda.ipynb` tagging tool: 8 heavy
  (40%), 7 clear (35%), 5 partial (25%). Occlusion is common enough in this sample to matter —
  consistent with the erasing-based augmentation step targeting exactly this gap.

## Augmentation pipeline

`src/data/augmentation.py` implements the training/validation transform pipelines
with `torchvision.transforms.v2`, differential per-class probabilities, and a
`WeightedRandomSampler`-ready weighting function. 11 unit tests cover shape/dtype correctness,
crop-wrapping behavior (minority applied unconditionally, majority wrapped at p=0.8), and
per-step probability differences between majority and minority classes.

- [notebooks/eda.ipynb](notebooks/eda.ipynb) — before/after augmented grids (majority pipeline,
  and a majority-vs-minority side-by-side with a fixed seed), plus a raw-count-vs-expected-draws
  chart showing the oversampling mechanism's effect on effective class balance

### Deviations from the original design

- **Crop scale widened from 0.7-1.0 to 0.4-1.0** after the gap analysis found a third of annotated
  animals occupy under 2% of frame area — the original range couldn't reach far enough to
  represent that tail during training.
- **Gaussian noise sigma is a fixed 0.02**, not a sampled 0.01-0.03 range — `torchvision`'s
  `GaussianNoise` doesn't support range sampling natively, so the pipeline uses the midpoint of
  the originally planned range instead.
- **Normalization uses ImageNet mean/std**, not dataset-computed statistics, matching the
  pretrained-backbone transfer-learning approach planned for classifier training.

These transforms are now wired into a `Dataset`/`DataLoader` in `src/classifier/dataset.py`,
consuming the localization stage's cropped-image outputs — see "Classifier training" below.

## Localization

`src/localization/` wraps MegaDetector v5a (via the `megadetector` package's `PTDetector`) as an
inference-only animal detector: `detector.py` handles model loading, inference, and
category/confidence filtering; `crop.py` expands and crops bounding boxes for classifier input;
`evaluate.py` computes IoU-based recall against ground-truth boxes. 13 unit tests cover the pure
geometry/filtering logic, plus one integration test against a real sample image and the real
downloaded model.

- `scripts/run_localization.py` — runs detection + cropping over the 633-image sample
  (`data/localization/detections.json`, `data/localization/crops/`, not committed)
- `scripts/generate_localization_report.py` — recall sanity check vs. ground truth
  (`reports/localization.md`)
- [notebooks/eda.ipynb](notebooks/eda.ipynb) — bounding boxes drawn on sample frames, plus the
  resulting animal crops

### Key findings

- **531 of 633 sample images** produced at least one animal detection (617 crops total) at the
  MegaDetector-documented "typical" confidence threshold (0.2).
- **92.5% recall** (331/358 ground-truth animals detected, IoU >= 0.5) on the 314 sample images
  that have ground-truth bounding-box annotations. This is a small-sample sanity check, not the
  full mAP/missed-detection-by-condition analysis planned for the dedicated evaluation stage —
  that needs real train/val/test splits to be meaningful.
- MegaDetector v5a was chosen over the newer "v1000" model family for better-documented, more
  widely benchmarked behavior — worth revisiting with a documented comparison if recall turns out
  to be a bottleneck once the classifier stage is running.

## Classifier training

`src/classifier/` labels each MegaDetector crop with a species (`labeling.py`, ground-truth IoU
matching where available, single-species fallback otherwise), splits crops into train/val/test
keeping near-duplicate source frames together (`split.py`), wraps them in a `Dataset` reusing the
augmentation pipeline (`dataset.py`), and provides a backbone factory (`models.py`, ResNet50 /
EfficientNet-B0 / ViT-B/16) plus class-weighted train/evaluate loop functions (`engine.py`). 22
unit tests cover the labeling rules, split grouping, dataset wiring, model output shapes, and
loss/loop correctness.

- `scripts/train_classifier.py` — labels crops, splits them, and trains + compares all three
  backbones with class-weighted cross-entropy and a `WeightedRandomSampler`, logging params and
  per-epoch metrics to local-file MLflow (`mlruns/`, not committed)
- [notebooks/eda.ipynb](notebooks/eda.ipynb) — per-backbone metrics pulled from MLflow and a
  final-accuracy comparison chart

### Key findings

- **594 crops labeled across 19 species** from the 617 MegaDetector crops (a handful dropped as
  genuinely ambiguous — multi-species source images with no ground-truth box to disambiguate
  which crop is which species). Split 416/90/88 (train/val/test), grouped so the 61 near-duplicate
  source-image pairs found during quality checks never land in different splits.
- **First 3-backbone comparison run** (5 epochs, ~500 training crops, class-weighted loss):
  ResNet50 14.4%, EfficientNet-B0 17.8%, ViT-B/16 38.9% final validation accuracy. ViT-B/16's lead
  is a real result of this specific run, now fully seed-locked (see "Reproducibility" below) — not
  a general claim about which architecture is "best" at this sample size. With single-digit
  per-species sample counts for several classes, some species appear in the training split only,
  and these numbers should be read as a pipeline-correctness sanity check (data flows correctly
  from crops through augmentation, model, class-weighted loss, to MLflow), not as a real
  performance benchmark. A full-dataset run is future work.

## Evaluation

`src/evaluation/` provides per-class classification metrics (`classifier_metrics.py`, thin
`scikit-learn` wrappers), day/night and per-site error segmentation (`segmentation.py`, pixel-based
day/night, not the unreliable hour proxy), and single-class detector average precision
(`detector_metrics.py`, a from-scratch precision-recall-integrated AP calculation).
`scripts/train_classifier.py` now persists each backbone's model weights and species-index mapping
as MLflow artifacts, so a trained model can be reloaded independently of the training run that
produced it.

- `scripts/evaluate_classifier.py` — loads the best backbone from the most recent training run,
  evaluates it on the 88-crop held-out test split: per-class precision/recall/F1, a confusion
  matrix (`reports/confusion_matrix.png`), day/night and per-site error breakdowns, and a
  qualitative list of every misclassified crop (`reports/classifier_evaluation.md`)
- `scripts/evaluate_detector.py` — average precision (IoU >= 0.5) against ground-truth boxes,
  plus missed-detection analysis by animal size and day/night (`reports/detector_evaluation.md`)
- [notebooks/eda.ipynb](notebooks/eda.ipynb) — the confusion matrix and both reports' raw text

### Key findings

- **Classifier: 42.0% test accuracy** (ViT-B/16, the best backbone from the latest training
  comparison), on 88 test crops across 19 species. Real per-class numbers vary widely by
  support — `pig` (1 test example) hits 100% recall, several other single-digit-support classes
  hit 0%, which is what per-class metrics genuinely look like at this sample size rather than a
  bug. Errors cluster heavily toward over-predicting `fox` and `squirrel` (visible directly in the
  confusion matrix), and night-time crops score noticeably worse than day (33.9% vs. 58.6%
  accuracy) — a real, plausible signal (IR grayscale captures lose the coat-color cues several
  species depend on) rather than a claim, given the small sample.
- **Detector: 0.535 Average Precision** (IoU >= 0.5) — meaningfully lower than the localization
  stage's 92.5% raw recall, because AP also penalizes false positives and integrates over the full
  precision-recall curve, including lower-confidence detections that recall-alone doesn't
  penalize. Per-box, IoU-matched detection rate by animal size: **large animals (>10% of frame)
  had the lowest detection rate (81.6%)**, ahead of small (88.2%), with medium animals (2-10%)
  detected 100% of the time — worth a closer qualitative look before assuming "small animals are
  the hard case," since this run's data says otherwise. Night detection rate (97.3%, per image)
  also slightly exceeds day (87.0%).
- **Occlusion segmentation is intentionally not included** in either evaluation: only 20 images
  have manual occlusion tags (from the dataset characterization stage's tagging tool), against an
  88-crop test set drawn from different source images — expected overlap is near zero, so a
  segmentation on that basis wouldn't be a real finding.

## Reproducibility

- **Environment**: `environment.yml` (committed) captures the exact conda + pip environment via
  `conda env export`, since most packages here were pip-installed (some with `--no-deps`) rather
  than conda-installed — `conda env create -f environment.yml` reproduces it exactly.
- **Seeds**: `train_classifier.py` locks `random`, `numpy`, and `torch`'s global seeds, sets
  `torch.backends.cudnn.deterministic = True` / `cudnn.benchmark = False` (cuDNN can otherwise pick
  different convolution algorithms run to run even with the seeds above locked), and passes an
  explicit seeded `torch.Generator` to `WeightedRandomSampler` rather than relying on global RNG
  state ordering.
- **Config, not inline settings**: per-script settings (learning rate, epochs, thresholds, sample
  sizes, seeds) live in `configs/*.yaml`, loaded via `src/utils/config.py`, not hardcoded constants
  — a run's exact settings are a diffable, versioned file, not buried in code.
- **Data versioning**: `scripts/generate_data_manifest.py` writes `reports/data_manifest.json`,
  checksumming (SHA-256) the annotation files individually and hashing each data directory
  (images, crops) as a combined digest — `data/` itself is gitignored (too large to commit), so
  this manifest is what lets a later reader verify "this run used exactly this data" without
  storing the data itself in git.
- **Automatic experiment metadata**: every `train_classifier.py` run logs its config file and (if
  present) the data manifest as MLflow artifacts, plus Python/PyTorch/CUDA versions as params,
  alongside the metrics it already logged — no manual bookkeeping needed to know what produced a
  given run's numbers later.
- A checksummed manifest was chosen over a dedicated tool like DVC, and plain YAML was chosen over
  a config framework like Hydra, to keep the reproducibility layer proportional to this project's
  scale — both are documented tradeoffs, not oversights.

## Tradeoffs

- Only annotation metadata was downloaded initially, not the full ~105GB image archive; a small
  stratified sample was pulled once image statistics and quality checks required real pixel data
  (633 images currently in `data/raw/images/`, sampled up to 20 per category including `empty`).
  This kept iteration fast without blocking on a multi-hour download, at the cost of quality-check
  and recall findings being sample-based rather than exhaustive until a full-dataset pass.
- MegaDetector (pretrained) is used for localization rather than training a custom detector, to
  keep engineering effort focused on classification, optimization, and deployment rather than
  object detection research.
- The classifier training split doesn't force hard per-class stratification across train/val/test
  — with per-species sample counts as low as single digits dataset-wide, that would either crash
  or require dropping already-scarce species entirely. The split keeps near-duplicate source
  images grouped instead, accepting that some rare species may be train-only as a known,
  documented consequence of the severe class imbalance found during characterization.

## Repository conventions

- Prefer small, focused files over large ones that do too much.
- Follow existing patterns before introducing new ones.
- Commit messages, pull request descriptions, code comments, and README content are written in
  first person, as the repository author.
