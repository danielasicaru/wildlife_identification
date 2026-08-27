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


def test_crop_dataset_passes_correct_is_minority_flag_per_row(tmp_path, monkeypatch):
    _write_sample_image(tmp_path / "crop0.jpg")
    _write_sample_image(tmp_path / "crop1.jpg")
    df = pd.DataFrame({"crop_file": ["crop0.jpg", "crop1.jpg"], "species": ["badger", "fox"]})

    calls = []
    real_build_train_transform = dataset_module.build_train_transform

    def spy(is_minority):
        calls.append(is_minority)
        return real_build_train_transform(is_minority=is_minority)

    monkeypatch.setattr(dataset_module, "build_train_transform", spy)

    dataset = CropDataset(
        df, crops_dir=tmp_path, species_to_index={"badger": 0, "fox": 1}, is_train=True, minority_species={"badger"}
    )
    dataset[0]  # badger -- minority
    dataset[1]  # fox -- not minority

    assert calls == [True, False]
