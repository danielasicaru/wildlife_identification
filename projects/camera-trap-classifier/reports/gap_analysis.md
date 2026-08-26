# Qualitative Gap Analysis — Scale Variation and Occlusion

## Scale variation

Annotated bounding boxes: 65112 (covering 63025 of the 243,100 total images -- bbox coverage is partial, not the full dataset)

Animal-area-to-image-area ratio: min=0.0001, 25th=0.0159, median=0.0293, 75th=0.0674, max=1.0000

Under 2% of frame area: 21878/65112 (33.6%)

Over 50% of frame area: 4316/65112 (6.6%)

**Finding**: the median animal occupies under 3% of the frame, with a third of annotated images under 2%. This is a more extreme long tail than a qualitative 'animals appear at different distances' statement suggests. **Follow-up**: revisit whether the augmentation spec's multi-scale crop range (0.7-1.0) adequately covers this tail, or whether a wider zoom-in range is needed to represent small, distant animals during training.

## Occlusion

Manually tagged sample: 20 images

- heavy: 8 (40.0%)
- clear: 7 (35.0%)
- partial: 5 (25.0%)
