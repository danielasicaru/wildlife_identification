"""Trains and compares backbones on a given labeled/split crop dataframe: class-weighted loss, a
WeightedRandomSampler, and early stopping with best-val_loss-epoch weight restoration. Shared by
the main training script and any alternative-split experiment (e.g. a site-disjoint split) so a
fix to this loop can't silently apply to only one of them."""
import json
import sys
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.classifier.dataset import CropDataset
from src.classifier.engine import EarlyStopping, compute_class_weights, evaluate, train_one_epoch
from src.classifier.models import build_model
from src.data.augmentation import build_sample_weights, minority_species


def train_and_compare_backbones(
    train_df, val_df, crops_dir: Path, species_to_index: dict[str, int], backbones: tuple[str, ...],
    seed: int, epochs: int, batch_size: int, learning_rate: float, early_stopping_patience: int,
    device: str, checkpoint_dir: Path, mlflow_params: dict, artifact_paths: list[Path],
) -> dict[str, dict]:
    """Returns {backbone: best_val_metrics}. Assumes the caller has already called
    mlflow.set_tracking_uri()/set_experiment(). `mlflow_params` are extra per-run params logged
    alongside the standard ones (e.g. which split strategy produced train_df/val_df);
    `artifact_paths` are extra files logged as MLflow artifacts (e.g. the config file used).
    """
    train_minority = minority_species(train_df["species"].value_counts())
    train_dataset = CropDataset(train_df, crops_dir, species_to_index, is_train=True, minority_species=train_minority)
    val_dataset = CropDataset(val_df, crops_dir, species_to_index, is_train=False)
    train_weights = build_sample_weights(train_df["species"], train_df["species"].value_counts())
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    class_weights = compute_class_weights(train_df["species"], species_to_index).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    results = {}
    for backbone in backbones:
        # Reseeded to `seed` per backbone so each one sees an identical batch order -- backbone is
        # the only thing that varies across iterations.
        sampler = WeightedRandomSampler(
            train_weights, num_samples=len(train_weights), replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        with mlflow.start_run(run_name=backbone):
            mlflow.log_params({
                "backbone": backbone, "seed": seed, "epochs": epochs,
                "batch_size": batch_size, "learning_rate": learning_rate,
                "train_size": len(train_df), "val_size": len(val_df), "num_classes": len(species_to_index),
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                **mlflow_params,
            })
            for artifact_path in artifact_paths:
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))

            model = build_model(backbone, num_classes=len(species_to_index)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            early_stopping = EarlyStopping(patience=early_stopping_patience, mode="min")

            best_state_dict = None
            best_val_metrics = None
            best_epoch = 0
            epochs_trained = 0
            for epoch in range(epochs):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_metrics = evaluate(model, val_loader, criterion, device)
                epochs_trained = epoch + 1
                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"]},
                    step=epoch,
                )
                print(
                    f"[{backbone}] epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f}"
                )

                should_stop = early_stopping.step(val_metrics["loss"])
                if early_stopping.epochs_without_improvement == 0:
                    # New best -- keep its weights (on CPU, so we're not holding two copies of the
                    # model on GPU at once) so a later, worse epoch doesn't overwrite the checkpoint.
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_val_metrics = val_metrics
                    best_epoch = epochs_trained

                if should_stop:
                    print(
                        f"[{backbone}] stopping early after {epochs_trained} epochs -- val_loss hasn't "
                        f"improved for {early_stopping_patience} consecutive epochs "
                        f"(best val_loss={early_stopping.best_score:.4f} at epoch {best_epoch})"
                    )
                    break

            model.load_state_dict(best_state_dict)
            results[backbone] = best_val_metrics
            mlflow.log_metrics({"final_val_accuracy": best_val_metrics["accuracy"]})
            mlflow.log_params({
                "epochs_trained": epochs_trained, "stopped_early": early_stopping.should_stop,
                "best_epoch": best_epoch,
            })

            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{backbone}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            mlflow.log_artifact(str(checkpoint_path))

            species_mapping_path = checkpoint_dir / "species_to_index.json"
            with open(species_mapping_path, "w", encoding="utf-8") as f:
                json.dump(species_to_index, f, indent=2)
            mlflow.log_artifact(str(species_mapping_path))

    return results
