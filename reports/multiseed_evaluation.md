# Multi-Seed Backbone Comparison

Each backbone trained and evaluated across 5 seeds (42, 43, 44, 45, 46), reseeding the train/val/test split (not just model init/sampler order) each time -- this captures split-sensitivity too, not just training-randomness, which matters at this dataset size. Numbers below are test accuracy per seed, not validation accuracy.

| backbone        |   n_seeds |   mean_test_accuracy |   std_test_accuracy |   min |   max |
|:----------------|----------:|---------------------:|--------------------:|------:|------:|
| vit_b_16        |         5 |                0.547 |               0.109 | 0.364 | 0.625 |
| resnet50        |         5 |                0.54  |               0.149 | 0.295 | 0.67  |
| efficientnet_b0 |         5 |                0.538 |               0.042 | 0.477 | 0.591 |

## Per-seed test accuracy

|   seed |   efficientnet_b0 |   resnet50 |   vit_b_16 |
|-------:|------------------:|-----------:|-----------:|
|     42 |             0.477 |      0.295 |      0.364 |
|     43 |             0.591 |      0.648 |      0.625 |
|     44 |             0.523 |      0.534 |      0.534 |
|     45 |             0.552 |      0.552 |      0.586 |
|     46 |             0.545 |      0.67  |      0.625 |