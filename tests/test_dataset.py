import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import src.classifier.dataset as dataset_module
from src.classifier.dataset import CropDataset


def _write_sample_image(path):
    arr = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype="uint8")
    Image.fromarray(arr).save(path)


def test_crop_dataset_returns_tensor_and_label_index(tmp_path):
    _write_sample_image(tmp_path / "crop0.jpg")
    df = pd.DataFrame({"crop_file": ["crop0.jpg"], "species": ["fox"]})
    species_to_index = {"fox": 0, "coyote": 1}

    dataset = CropDataset(df, crops_dir=tmp_path, species_to_index=species_to_index, is_train=False)
    image, label = dataset[0]

    assert image.shape == (3, 224, 224)
    assert label == 0


def test_crop_dataset_length_matches_dataframe():
    df = pd.DataFrame({"crop_file": ["a.jpg", "b.jpg"], "species": ["fox", "coyote"]})

    dataset = CropDataset(df, crops_dir=".", species_to_index={"fox": 0, "coyote": 1}, is_train=False, minority_species=set())

    assert len(dataset) == 2


def test_crop_dataset_builds_both_transforms_once_at_init(tmp_path, monkeypatch):
    df = pd.DataFrame({"crop_file": ["crop0.jpg", "crop1.jpg"], "species": ["badger", "fox"]})

    calls = []

    def spy(is_minority):
        calls.append(is_minority)
        return lambda img: f"transform(minority={is_minority})"

    monkeypatch.setattr(dataset_module, "build_train_transform", spy)

    CropDataset(df, crops_dir=tmp_path, species_to_index={"badger": 0, "fox": 1}, is_train=True, minority_species={"badger"})

    # Built exactly once each at construction time, not once per __getitem__ call.
    assert calls == [True, False]


def test_crop_dataset_selects_correct_precomputed_transform_per_row(tmp_path, monkeypatch):
    _write_sample_image(tmp_path / "crop0.jpg")
    _write_sample_image(tmp_path / "crop1.jpg")
    df = pd.DataFrame({"crop_file": ["crop0.jpg", "crop1.jpg"], "species": ["badger", "fox"]})

    monkeypatch.setattr(
        dataset_module, "build_train_transform", lambda is_minority: (lambda img: f"minority={is_minority}")
    )

    dataset = CropDataset(
        df, crops_dir=tmp_path, species_to_index={"badger": 0, "fox": 1}, is_train=True, minority_species={"badger"}
    )

    image0, _ = dataset[0]  # badger -- minority
    image1, _ = dataset[1]  # fox -- not minority

    assert image0 == "minority=True"
    assert image1 == "minority=False"
