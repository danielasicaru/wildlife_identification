# Hyperparameter Search

Small grid search over learning rate and batch size on **efficientnet_b0** only, one seed (42) each, same train/val split for every grid point (only the hyperparameters vary). The multi-seed backbone comparison already showed backbone choice barely matters at this dataset size, so this asks a narrower, cheaper question: was the value already in `configs/train_classifier.yaml` a good pick, or just the first thing tried.

|   learning_rate |   batch_size |   val_accuracy |
|----------------:|-------------:|---------------:|
|          0.0001 |           16 |          0.5   |
|          0.0005 |           32 |          0.444 |
|          5e-05  |           16 |          0.444 |
|          0.0005 |           16 |          0.367 |
|          5e-05  |           32 |          0.311 |
|          0.0001 |           32 |          0.267 |
