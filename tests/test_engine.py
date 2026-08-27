import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.classifier.engine import compute_class_weights, evaluate, train_one_epoch


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
