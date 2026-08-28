# Classifier Evaluation

Best run (latest batch): **efficientnet_b0** (val_accuracy=0.511)
Test set: 88 crops across 19 species
Overall test accuracy: **47.7%**

## Per-class metrics

|               |   precision |   recall |   f1-score |   support |
|:--------------|------------:|---------:|-----------:|----------:|
| badger        |       0.429 |    0.6   |      0.5   |         5 |
| bird          |       0.2   |    0.143 |      0.167 |         7 |
| bobcat        |       0     |    0     |      0     |         0 |
| cat           |       0     |    0     |      0     |         7 |
| cow           |       0.667 |    0.5   |      0.571 |         4 |
| coyote        |       0.333 |    0.167 |      0.222 |         6 |
| deer          |       0.7   |    0.636 |      0.667 |        11 |
| dog           |       0.714 |    0.714 |      0.714 |         7 |
| fox           |       0     |    0     |      0     |         1 |
| insect        |       0     |    0     |      0     |         0 |
| lizard        |       1     |    0.333 |      0.5   |         3 |
| mountain_lion |       0.5   |    1     |      0.667 |         2 |
| opossum       |       0.583 |    1     |      0.737 |         7 |
| pig           |       1     |    1     |      1     |         1 |
| rabbit        |       1     |    0.125 |      0.222 |         8 |
| raccoon       |       0.667 |    0.25  |      0.364 |         8 |
| rodent        |       0.667 |    0.8   |      0.727 |         5 |
| skunk         |       0.5   |    1     |      0.667 |         3 |
| squirrel      |       0.333 |    0.667 |      0.444 |         3 |

## Confusion matrix

See `reports/confusion_matrix.png`.

## Day/night segmentation

| day_night   |   mean |   count |
|:------------|-------:|--------:|
| day         |  0.483 |      29 |
| night       |  0.475 |      59 |

## Per-site errors (raw counts, sample too fragmented for accuracy-rate claims)

27 sites had at least one misclassification, out of 42 sites represented in the 88-crop test set (most sites have 1-7 test examples -- too few for a reliable per-site accuracy rate).

## Failure analysis (qualitative)

- `585f4fd4-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rodent, predicted=skunk (confidence=0.13)
- `5860ef79-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=lizard, predicted=bird (confidence=0.10)
- `5860f084-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cow, predicted=bird (confidence=0.25)
- `586291fd-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=bird, predicted=bobcat (confidence=0.18)
- `5865e535-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=skunk (confidence=0.20)
- `5865e535-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=cat, predicted=raccoon (confidence=0.16)
- `58782b2c-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=squirrel (confidence=0.74)
- `58823be4-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=rodent (confidence=0.24)
- `588718dd-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=bird, predicted=squirrel (confidence=0.45)
- `5888be7a-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=deer, predicted=cat (confidence=0.19)
- `5897b282-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=raccoon, predicted=opossum (confidence=0.31)
- `5897b282-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=raccoon, predicted=rodent (confidence=0.21)
- `58b1317b-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=badger (confidence=0.55)
- `58b66581-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=fox, predicted=coyote (confidence=0.46)
- `58b82279-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=lizard, predicted=dog (confidence=0.45)
- `58c34d65-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=deer (confidence=0.18)
- `58c97ba9-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=deer, predicted=coyote (confidence=0.20)
- `58d7a448-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=badger (confidence=0.24)
- `58dc4b16-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=raccoon, predicted=opossum (confidence=0.20)
- `58e40e9b-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=deer, predicted=mountain_lion (confidence=0.70)
- `58e40e9b-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=deer, predicted=fox (confidence=0.80)
- `58ede426-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=fox (confidence=0.47)
- `58fbecb7-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=bird, predicted=squirrel (confidence=0.15)
- `590a0e87-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cow, predicted=badger (confidence=0.26)
- `5911d9d6-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=coyote, predicted=bobcat (confidence=0.26)
- `59136598-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=fox (confidence=0.18)
- `592abf28-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=dog, predicted=mountain_lion (confidence=0.45)
- `59439e70-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=badger, predicted=fox (confidence=0.14)
- `59439e70-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=badger, predicted=skunk (confidence=0.55)
- `59484770-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=coyote, predicted=deer (confidence=0.43)
- `596106a0-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=bird, predicted=dog (confidence=0.40)
- `596ef0f3-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=bird, predicted=squirrel (confidence=0.38)
- `5971f8de-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=bird, predicted=deer (confidence=0.14)
- `59817a1e-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=opossum (confidence=0.33)
- `59817a5b-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=coyote, predicted=badger (confidence=0.40)
- `59862370-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=bobcat (confidence=0.56)
- `598623f9-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=squirrel, predicted=bird (confidence=0.24)
- `5998cc14-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=bobcat (confidence=0.28)
- `59bfddb2-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=coyote, predicted=fox (confidence=0.15)
- `59cf4519-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=coyote, predicted=bobcat (confidence=0.24)
- `59cf45a2-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=cat, predicted=opossum (confidence=0.22)
- `59d75237-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=raccoon, predicted=opossum (confidence=0.15)
- `59d75237-23d2-11e8-a6a3-ec086b02610b_crop1.jpg`: true=raccoon, predicted=fox (confidence=0.32)
- `59eab753-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=raccoon, predicted=bird (confidence=0.31)
- `5a1e52a5-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=rabbit, predicted=cow (confidence=0.40)
- `5a1fe8f6-23d2-11e8-a6a3-ec086b02610b_crop0.jpg`: true=dog, predicted=bobcat (confidence=0.55)

Occlusion segmentation is not included: only 20 images have manual occlusion tags (from the dataset characterization stage), against 88 test crops from different source images -- expected overlap is near zero, so a segmentation on that basis wouldn't be meaningful at this sample size.