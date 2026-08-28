import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.classifier.engine import EarlyStopping, compute_class_weights, evaluate, train_one_epoch


def test_compute_class_weights_inverse_frequency_ordered_by_index():
    labels = pd.Series(["fox", "fox", "coyote"])
    species_to_index = {"coyote": 0, "fox": 1}

    weights = compute_class_weights(labels, species_to_index)

    assert weights.shape == (2,)
    assert weights[0] > weights[1]


def test_compute_class_weights_defaults_to_one_for_species_absent_from_labels():
    labels = pd.Series(["fox", "fox"])
    species_to_index = {"fox": 0, "badger": 1}  # badger has zero rows in labels

    weights = compute_class_weights(labels, species_to_index)

    assert weights[1] == pytest.approx(1.0)


def test_train_one_epoch_returns_finite_loss():
    x = torch.randn(8, 3, 4, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    loss = train_one_epoch(model, loader, optimizer, criterion, device="cpu")

    assert loss == loss
    assert loss > 0


def test_evaluate_returns_loss_and_accuracy():
    x = torch.randn(8, 3, 4, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    criterion = nn.CrossEntropyLoss()

    result = evaluate(model, loader, criterion, device="cpu")

    assert "loss" in result
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_early_stopping_does_not_stop_while_score_keeps_improving():
    stopper = EarlyStopping(patience=3, mode="min")

    for loss in [1.0, 0.8, 0.6, 0.4]:
        should_stop = stopper.step(loss)
        assert should_stop is False


def test_early_stopping_stops_after_patience_epochs_without_improvement():
    stopper = EarlyStopping(patience=3, mode="min")

    stopper.step(1.0)  # best so far
    assert stopper.step(1.1) is False  # 1 epoch without improvement
    assert stopper.step(1.2) is False  # 2 epochs without improvement
    assert stopper.step(0.9) is False  # improved again, counter resets
    assert stopper.step(1.0) is False  # 1
    assert stopper.step(1.0) is False  # 2
    assert stopper.step(1.0) is True  # 3 -- patience exhausted


def test_early_stopping_max_mode_for_accuracy():
    stopper = EarlyStopping(patience=2, mode="max")

    stopper.step(0.5)  # best so far
    assert stopper.step(0.4) is False  # 1 epoch without improvement
    assert stopper.step(0.6) is False  # improved, counter resets
    assert stopper.step(0.6) is False  # not strictly better, 1
    assert stopper.step(0.6) is True  # 2 -- patience exhausted


def test_early_stopping_tracks_best_score():
    stopper = EarlyStopping(patience=5, mode="min")

    stopper.step(1.0)
    stopper.step(0.5)
    stopper.step(0.8)

    assert stopper.best_score == pytest.approx(0.5)


def test_early_stopping_min_delta_ignores_negligible_improvement():
    stopper = EarlyStopping(patience=2, mode="min", min_delta=0.01)

    stopper.step(1.0)  # best so far
    assert stopper.step(0.999) is False  # improvement smaller than min_delta -- doesn't count, 1
    assert stopper.step(0.998) is True  # still below min_delta -- 2, patience exhausted


def test_early_stopping_min_delta_still_resets_on_real_improvement():
    stopper = EarlyStopping(patience=2, mode="min", min_delta=0.01)

    stopper.step(1.0)
    stopper.step(0.999)  # negligible, 1 epoch without improvement
    assert stopper.step(0.9) is False  # real improvement (>= min_delta) -- resets counter
    assert stopper.best_score == pytest.approx(0.9)
