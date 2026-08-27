"""Near-duplicate-aware grouped train/val/test split. See Task 2 design note in the plan for why
this doesn't attempt hard per-class stratification."""
import random

import pandas as pd


def group_images_by_near_duplicates(images: list[str], duplicate_pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Union-find grouping: images connected (directly or transitively) by a near-duplicate pair
    get the same group id. Ungrouped images each get their own singleton group.
    """
    parent = {image: image for image in images}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for a, b in duplicate_pairs:
        if a in parent and b in parent:
            union(a, b)

    return {image: find(image) for image in images}


def split_groups(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42) -> pd.Series:
    """Assigns each row to train/val/test by shuffling unique group_id values (seeded) and
    greedily filling splits by row-count target fractions, keeping every group intact.
    """
    group_sizes = df.groupby("group_id").size()
    groups = list(group_sizes.index)
    random.Random(seed).shuffle(groups)

    total = len(df)
    train_target = train_frac * total
    val_target = val_frac * total

    assignment = {}
    train_count = val_count = 0
    for group_id in groups:
        size = group_sizes[group_id]
        if train_count < train_target:
            assignment[group_id] = "train"
            train_count += size
        elif val_count < val_target:
            assignment[group_id] = "val"
            val_count += size
        else:
            assignment[group_id] = "test"

    return df["group_id"].map(assignment)
