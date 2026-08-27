"""torch.utils.data.Dataset wrapping labeled crop rows, reusing src.data.augmentation's
transform pipelines."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.augmentation import build_train_transform, build_val_transform


class CropDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        crops_dir,
        species_to_index: dict[str, int],
        is_train: bool,
        minority_species: set[str] | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.crops_dir = Path(crops_dir)
        self.species_to_index = species_to_index
        self.is_train = is_train
        self.minority_species = minority_species or set()
        self._val_transform = build_val_transform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = Image.open(self.crops_dir / row["crop_file"]).convert("RGB")

        if self.is_train:
            transform = build_train_transform(is_minority=row["species"] in self.minority_species)
        else:
            transform = self._val_transform

        return transform(image), self.species_to_index[row["species"]]
