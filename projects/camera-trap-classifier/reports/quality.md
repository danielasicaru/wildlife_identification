# Quality Checks — Caltech Camera Traps (sample)

Sample size: 633 images (stratified, ~20 per category)

## Corruption

Corrupted: 0/633

## Blur

Laplacian variance: min=75.4, max=2513.3, median=331.6

Flagged as blur candidates (bottom 10%, threshold=255.9): 64/633

## Color channel / day-night proxy validation

Effectively grayscale (real pixel check, tolerance=2.0): 379/633 (59.9%)

Hour-based day/night proxy (computed from annotations) estimated 40.0% night.

Gap between pixel-based and hour-based estimates: 19.9 percentage points.

## Near-duplicates

Near-duplicate pairs found (perceptual hash, Hamming distance <= 5): 61

## Findings and recommended follow-up

- **Corruption**: none found in this sample. Not a concern for this dataset at this sample size; worth a full-dataset pass before final training, not before.
- **Blur**: real spread exists (see min/max above). Recommend visually spot-checking the flagged bottom-decile images before deciding whether to exclude them or keep them as intentionally hard examples -- camera traps trigger on motion, so some blur may be an inherent, unavoidable part of the true data distribution, not noise to remove.
- **Day/night proxy accuracy**: the 20-point gap between the pixel-based grayscale check and the hour-based proxy means the hour proxy is not reliable enough to use as a ground-truth label. Recommend replacing the proxy with the pixel-based `is_effectively_grayscale` check directly wherever day/night matters downstream (e.g. for the selective grayscale augmentation's targeting logic), rather than widening the day/night hour window to try to match it.
- **Near-duplicates**: found in this sample -- recommend deduplicating before creating train/val/test splits, so near-identical frames don't leak across splits and inflate validation metrics.
