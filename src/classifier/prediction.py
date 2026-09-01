"""Runs a trained classifier over a labeled crop dataframe and returns per-crop predictions.
Shared by every evaluation script (primary, site-holdout, multi-seed) so the prediction loop
itself can't drift between them."""
import pandas as pd
import torch
import torch.nn as nn

from src.classifier.dataset import CropDataset


def predict_test_set(
    model: nn.Module, test_df: pd.DataFrame, crops_dir, species_to_index: dict[str, int],
    index_to_species: dict[int, str], device: str,
) -> pd.DataFrame:
    """Returns a DataFrame with columns [crop_file, true, predicted, confidence], one row per crop
    in test_df, in the same order. `model` must already be in eval mode.
    """
    test_dataset = CropDataset(test_df, crops_dir, species_to_index, is_train=False)
    rows = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, label_index = test_dataset[i]
            row = test_df.iloc[i]
            logits = model(image.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)
            pred_index = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, pred_index].item())

            rows.append({
                "crop_file": row["crop_file"],
                "true": index_to_species[label_index],
                "predicted": index_to_species[pred_index],
                "confidence": confidence,
            })

    return pd.DataFrame(rows, columns=["crop_file", "true", "predicted", "confidence"])
