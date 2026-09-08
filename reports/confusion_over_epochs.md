# Per-Class Accuracy Over Training Epochs

Validation-set per-class accuracy (recall) tracked at every epoch during training for **efficientnet_b0** (this run's best backbone), not just a final-epoch snapshot -- shows which classes' errors are stable, improving, or degrading as training progresses. The validation set is small (90 crops across 19 species, many classes with single-digit support), so per-class accuracy swings by 50 percentage points on a single flipped prediction -- read trends here as directional, not precise.

37 epochs tracked. Accuracy change from the first half of training to the second half (second-half mean minus first-half mean; negative = got worse as training progressed):

|               |   accuracy_change |
|:--------------|------------------:|
| lizard        |            -0.561 |
| pig           |             0     |
| insect        |             0     |
| mountain_lion |             0     |
| rabbit        |             0     |
| fox           |             0.014 |
| squirrel      |             0.094 |
| cat           |             0.132 |
| cow           |             0.148 |
| badger        |             0.164 |
| raccoon       |             0.184 |
| opossum       |             0.224 |
| bobcat        |             0.26  |
| skunk         |             0.315 |
| bird          |             0.316 |
| deer          |             0.378 |
| coyote        |             0.461 |
| rodent        |             0.497 |
| dog           |             0.513 |

Full per-epoch, per-class accuracy for every backbone: `reports/confusion_over_epochs_<backbone>.csv`.