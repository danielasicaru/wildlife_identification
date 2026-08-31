# Site-Holdout Generalization Check

Same crops and labeling as the main classifier evaluation, but split so every crop from a given camera site lands entirely in train, val, or test -- no site appears in more than one split. Test accuracy here measures generalization to camera sites the model never saw during training, as opposed to reports/classifier_evaluation.md's split, which only guarantees unseen images (the same site can appear in both train and test there).

Best run (latest site-holdout batch): **resnet50** (val_accuracy=0.511)
Test set: 85 crops across 19 species, from 12 camera sites (train: 419 crops / 64 sites, val: 90 crops / 17 sites).
Overall test accuracy: **38.8%**

## Per-class metrics

|               |   precision |   recall |   f1-score |   support |
|:--------------|------------:|---------:|-----------:|----------:|
| badger        |       0.25  |    0.25  |      0.25  |         4 |
| bird          |       0.933 |    0.737 |      0.824 |        19 |
| bobcat        |       0     |    0     |      0     |         3 |
| cat           |       0     |    0     |      0     |         0 |
| cow           |       0.5   |    0.091 |      0.154 |        11 |
| coyote        |       0.667 |    0.5   |      0.571 |         8 |
| deer          |       0.25  |    0.5   |      0.333 |         4 |
| dog           |       0     |    0     |      0     |         1 |
| fox           |       0     |    0     |      0     |         4 |
| insect        |       0     |    0     |      0     |         0 |
| lizard        |       0     |    0     |      0     |         0 |
| mountain_lion |       0     |    0     |      0     |         3 |
| opossum       |       0.125 |    1     |      0.222 |         1 |
| pig           |       0     |    0     |      0     |         0 |
| rabbit        |       0.625 |    0.833 |      0.714 |         6 |
| raccoon       |       0     |    0     |      0     |        10 |
| rodent        |       0.8   |    0.444 |      0.571 |         9 |
| skunk         |       0     |    0     |      0     |         1 |
| squirrel      |       0.2   |    1     |      0.333 |         1 |