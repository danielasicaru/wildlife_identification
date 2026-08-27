"""Class-weighted loss setup and epoch-level train/evaluate loops, backbone-agnostic."""
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_class_weights(labels: pd.Series, species_to_index: dict[str, int]) -> torch.Tensor:
    """Inverse-class-frequency weights, ordered by species_to_index, for CrossEntropyLoss(weight=...).

    Complements (not substitutes for) the augmentation pipeline's oversampling/differential-
    probability imbalance handling -- this operates at the loss level.
    """
    counts = labels.value_counts()
    weights = torch.zeros(len(species_to_index))
    for species, index in species_to_index.items():
        weights[index] = 1.0 / counts.get(species, 1)
    return weights


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: str) -> float:
    model.train()
    model.to(device)
    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: str) -> dict:
    model.eval()
    model.to(device)
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            n_batches += 1
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return {"loss": total_loss / n_batches, "accuracy": correct / total}
