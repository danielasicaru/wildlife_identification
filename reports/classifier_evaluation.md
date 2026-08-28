# Classifier Evaluation

Best run (latest batch): **vit_b_16** (val_accuracy=0.600)
Test set: 88 crops across 19 species
Overall test accuracy: **52.3%**

## Per-class metrics

|               |   precision |   recall |   f1-score |   support |
|:--------------|------------:|---------:|-----------:|----------:|
| badger        |       0.75  |    0.6   |      0.667 |         5 |
| bird          |       1     |    0.286 |      0.444 |         7 |
| bobcat        |       0     |    0     |      0     |         0 |
| cat           |       0.5   |    0.143 |      0.222 |         7 |
| cow           |       0.5   |    0.5   |      0.5   |         4 |
| coyote        |       0.5   |    0.167 |      0.25  |         6 |
| deer          |       0.538 |    0.636 |      0.583 |        11 |
| dog           |       0.667 |    0.857 |      0.75  |         7 |
| fox           |       0     |    0     |      0     |         1 |
| insect        |       0     |    0     |      0     |         0 |
| lizard        |       1     |    0.333 |      0.5   |         3 |
| mountain_lion |       0.222 |    1     |      0.364 |         2 |
| opossum       |       0.667 |    0.571 |      0.615 |         7 |
| pig           |       1     |    1     |      1     |         1 |
| rabbit        |       0.75  |    0.375 |      0.5   |         8 |
| raccoon       |       0.545 |    0.75  |      0.632 |         8 |
| rodent        |       0.667 |    0.8   |      0.727 |         5 |
| skunk         |       0.667 |    0.667 |      0.667 |         3 |
| squirrel      |       0.25  |    0.333 |      0.286 |         3 |

## Confusion matrix

See `reports/confusion_matrix.png`.

## Day/night segmentation

| day_night   |   mean |   count |
|:------------|-------:|--------:|
| day         |  0.552 |      29 |
| night       |  0.508 |      59 |

## Per-site errors (raw counts, sample too fragmented for accuracy-rate claims)

23 sites had at least one misclassification, out of 42 sites represented in the 88-crop test set (most sites have 1-7 test examples -- too few for a reliable per-site accuracy rate).

## Failure analysis (qualitative)

- `585f4fd4-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rodent, predicted=opossum (confidence=0.36)
- `5860ef79-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=lizard, predicted=deer (confidence=0.52)
- `5860f084-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cow, predicted=coyote (confidence=0.18)
- `5865e535-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=skunk (confidence=0.48)
- `5865e535-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=cat, predicted=raccoon (confidence=0.61)
- `58782b2c-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=squirrel (confidence=0.89)
- `588718dd-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=bird, predicted=squirrel (confidence=0.29)
- `5888be7a-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=deer, predicted=mountain_lion (confidence=0.67)
- `5888be7a-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=deer, predicted=cow (confidence=0.94)
- `5897b282-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=raccoon, predicted=rodent (confidence=0.50)
- `58b1317b-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=fox (confidence=0.80)
- `58b66581-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=fox, predicted=deer (confidence=0.35)
- `58b82279-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=lizard, predicted=dog (confidence=0.44)
- `58d7a448-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=insect (confidence=0.31)
- `58e40e9b-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=deer, predicted=mountain_lion (confidence=0.65)
- `58e40e9b-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=deer, predicted=cow (confidence=0.92)
- `58ede426-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=fox (confidence=0.39)
- `58fbecb7-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=bird, predicted=raccoon (confidence=0.41)
- `590a0e87-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cow, predicted=badger (confidence=0.99)
- `5911d9d6-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=coyote, predicted=deer (confidence=0.85)
- `59136598-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=fox (confidence=0.34)
- `59439e70-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=badger, predicted=rabbit (confidence=0.34)
- `59439e70-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=badger, predicted=fox (confidence=0.82)
- `59484770-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=coyote, predicted=deer (confidence=0.67)
- `596106a0-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=bird, predicted=dog (confidence=0.75)
- `596ef0f3-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=bird, predicted=squirrel (confidence=0.72)
- `5971f8de-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=bird, predicted=deer (confidence=0.38)
- `5979bc53-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=opossum, predicted=mountain_lion (confidence=0.39)
- `59817a1e-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=rodent (confidence=0.89)
- `59817a5b-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=coyote, predicted=mountain_lion (confidence=0.61)
- `59862370-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=raccoon (confidence=0.86)
- `598623f9-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=squirrel, predicted=bobcat (confidence=0.57)
- `59a49bf0-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=skunk, predicted=deer (confidence=0.20)
- `59adfdb0-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=opossum, predicted=mountain_lion (confidence=0.31)
- `59be4165-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=squirrel, predicted=raccoon (confidence=0.38)
- `59bfddb2-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=coyote, predicted=raccoon (confidence=0.42)
- `59cf4519-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=coyote, predicted=bobcat (confidence=0.86)
- `59cf45a2-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=opossum (confidence=0.56)
- `59eab753-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=raccoon, predicted=mountain_lion (confidence=0.27)
- `5a1e52a5-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=mountain_lion (confidence=0.42)
- `5a1fe8f6-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=dog, predicted=cat (confidence=0.41)
- `5a27d7e0-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=opossum, predicted=dog (confidence=0.35)

Occlusion segmentation is not included: only 20 images have manual occlusion tags (from the dataset characterization stage), against 88 test crops from different source images -- expected overlap is near zero, so a segmentation on that basis wouldn't be meaningful at this sample size.