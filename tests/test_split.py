import pandas as pd
import pytest

from src.classifier.split import group_images_by_near_duplicates, split_groups


def test_group_images_by_near_duplicates_merges_pairs():
    images = ["a.jpg", "b.jpg", "c.jpg"]
    duplicate_pairs = [("a.jpg", "b.jpg")]

    groups = group_images_by_near_duplicates(images, duplicate_pairs)

    assert groups["a.jpg"] == groups["b.jpg"]
    assert groups["c.jpg"] != groups["a.jpg"]


def test_group_images_by_near_duplicates_transitive_chain():
    images = ["a.jpg", "b.jpg", "c.jpg"]
    duplicate_pairs = [("a.jpg", "b.jpg"), ("b.jpg", "c.jpg")]

    groups = group_images_by_near_duplicates(images, duplicate_pairs)

    assert groups["a.jpg"] == groups["b.jpg"] == groups["c.jpg"]


def test_group_images_by_near_duplicates_no_pairs_gives_singleton_groups():
    images = ["a.jpg", "b.jpg"]

    groups = group_images_by_near_duplicates(images, [])

    assert groups["a.jpg"] != groups["b.jpg"]


def test_split_groups_keeps_duplicate_group_together():
    df = pd.DataFrame({
        "crop_file": ["c0", "c1", "c2", "c3"],
        "source_image": ["a.jpg", "a.jpg", "b.jpg", "c.jpg"],
        "species": ["fox", "fox", "deer", "coyote"],
        "group_id": ["g1", "g1", "g2", "g3"],
    })

    result = split_groups(df, seed=42)

    g1_splits = result[df["group_id"] == "g1"].unique()
    assert len(g1_splits) == 1


def test_split_groups_returns_only_valid_split_names():
    df = pd.DataFrame({
        "crop_file": [f"c{i}" for i in range(20)],
        "source_image": [f"img{i}.jpg" for i in range(20)],
        "species": ["fox"] * 20,
        "group_id": [f"g{i}" for i in range(20)],
    })

    result = split_groups(df, seed=42)

    assert set(result.unique()) <= {"train", "val", "test"}


def test_split_groups_is_deterministic_for_fixed_seed():
    df = pd.DataFrame({
        "crop_file": [f"c{i}" for i in range(20)],
        "source_image": [f"img{i}.jpg" for i in range(20)],
        "species": ["fox"] * 20,
        "group_id": [f"g{i}" for i in range(20)],
    })

    result_a = split_groups(df, seed=42)
    result_b = split_groups(df, seed=42)

    assert (result_a.values == result_b.values).all()
