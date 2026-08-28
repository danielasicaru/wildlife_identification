# Classifier Evaluation

Best run (latest batch): **vit_b_16** (val_accuracy=0.389)
Test set: 88 crops across 19 species
Overall test accuracy: **42.0%**

## Per-class metrics

|               |   precision |   recall |   f1-score |   support |
|:--------------|------------:|---------:|-----------:|----------:|
| badger        |       0.444 |    0.8   |      0.571 |         5 |
| bird          |       0.833 |    0.714 |      0.769 |         7 |
| bobcat        |       0     |    0     |      0     |         0 |
| cat           |       0     |    0     |      0     |         7 |
| cow           |       0.667 |    0.5   |      0.571 |         4 |
| coyote        |       0     |    0     |      0     |         6 |
| deer          |       1     |    0.545 |      0.706 |        11 |
| dog           |       1     |    0.429 |      0.6   |         7 |
| fox           |       0.059 |    1     |      0.111 |         1 |
| insect        |       0     |    0     |      0     |         0 |
| lizard        |       0     |    0     |      0     |         3 |
| mountain_lion |       1     |    0.5   |      0.667 |         2 |
| opossum       |       0.5   |    0.143 |      0.222 |         7 |
| pig           |       1     |    1     |      1     |         1 |
| rabbit        |       0.5   |    0.375 |      0.429 |         8 |
| raccoon       |       0.167 |    0.125 |      0.143 |         8 |
| rodent        |       0.8   |    0.8   |      0.8   |         5 |
| skunk         |       1     |    0.667 |      0.8   |         3 |
| squirrel      |       0.188 |    1     |      0.316 |         3 |

## Confusion matrix

See `reports/confusion_matrix.png`.

## Day/night segmentation

| day_night   |   mean |   count |
|:------------|-------:|--------:|
| day         |  0.586 |      29 |
| night       |  0.339 |      59 |

## Per-site errors (raw counts, sample too fragmented for accuracy-rate claims)

27 sites had at least one misclassification, out of 42 sites represented in the 88-crop test set (most sites have 1-7 test examples -- too few for a reliable per-site accuracy rate).

## Failure analysis (qualitative)

- `585f4fd4-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rodent, predicted=raccoon (confidence=0.12)
- `5860ef79-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=lizard, predicted=squirrel (confidence=0.85)
- `5860f084-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cow, predicted=squirrel (confidence=0.32)
- `5865e535-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=fox (confidence=0.28)
- `5865e535-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=cat, predicted=raccoon (confidence=0.17)
- `58782b2c-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=squirrel (confidence=0.66)
- `58823be4-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=squirrel (confidence=0.29)
- `5888be7a-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=deer, predicted=fox (confidence=0.20)
- `5897b282-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=raccoon, predicted=rabbit (confidence=0.45)
- `5897b282-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=raccoon, predicted=fox (confidence=0.22)
- `58a0232a-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=mountain_lion, predicted=fox (confidence=0.38)
- `58af7579-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=opossum, predicted=raccoon (confidence=0.20)
- `58b1317b-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=fox (confidence=0.41)
- `58b82279-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=lizard, predicted=squirrel (confidence=0.22)
- `58c97ba9-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=deer, predicted=cat (confidence=0.19)
- `58dc4b16-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=raccoon, predicted=opossum (confidence=0.23)
- `58e40e9b-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=deer, predicted=fox (confidence=0.34)
- `58e40e9b-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=deer, predicted=cow (confidence=0.29)
- `58ede426-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=fox (confidence=0.28)
- `58ff0f3d-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=coyote, predicted=fox (confidence=0.76)
- `590a0e87-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cow, predicted=badger (confidence=0.96)
- `590d2a5c-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=lizard, predicted=squirrel (confidence=0.63)
- `5911d9d6-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=coyote, predicted=fox (confidence=0.76)
- `59136598-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=fox (confidence=0.43)
- `592abf28-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=dog, predicted=squirrel (confidence=0.40)
- `592abf28-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=dog, predicted=squirrel (confidence=0.25)
- `592c4f1e-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=raccoon, predicted=squirrel (confidence=0.22)
- `59439e70-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=badger, predicted=cat (confidence=0.18)
- `59484770-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=coyote, predicted=fox (confidence=0.40)
- `596106a0-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=bird, predicted=squirrel (confidence=0.32)
- `596d63a1-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=deer, predicted=fox (confidence=0.54)
- `596ef0f3-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=bird, predicted=squirrel (confidence=0.49)
- `5979bc53-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=opossum, predicted=badger (confidence=0.13)
- `59817a1e-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=rodent (confidence=0.19)
- `59817a5b-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=coyote, predicted=badger (confidence=0.41)
- `59862370-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=bobcat (confidence=0.41)
- `598de9ee-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=opossum, predicted=badger (confidence=0.54)
- `5998cc14-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=bobcat (confidence=0.24)
- `59a49b7d-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=opossum, predicted=fox (confidence=0.22)
- `59a49bf0-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=skunk, predicted=fox (confidence=0.33)
- `59adfdb0-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=opossum, predicted=badger (confidence=0.11)
- `59bfddb2-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=coyote, predicted=rabbit (confidence=0.22)
- `59cf4519-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=coyote, predicted=fox (confidence=0.57)
- `59cf45a2-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=raccoon (confidence=0.11)
- `59d75237-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=raccoon, predicted=rabbit (confidence=0.33)
- `59d75237-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=raccoon, predicted=bobcat (confidence=0.20)
- `59e77e26-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=dog, predicted=squirrel (confidence=0.28)
- `59eab753-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=raccoon, predicted=bird (confidence=0.21)
- `5a1e52a5-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=squirrel (confidence=0.62)
- `5a1fe8f6-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=dog, predicted=fox (confidence=0.32)
- `5a27d7e0-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=opossum, predicted=raccoon (confidence=0.15)

Occlusion segmentation is not included: only 20 images have manual occlusion tags (from the dataset characterization stage), against 88 test crops from different source images -- expected overlap is near zero, so a segmentation on that basis wouldn't be meaningful at this sample size.