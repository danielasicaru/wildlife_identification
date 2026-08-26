import pandas as pd
import pytest

from src.data.augmentation import build_sample_weights, minority_species


def test_minority_species_below_median_count():
    counts = pd.Series({"bobcat": 100, "coyote": 50, "fox": 10, "badger": 5}, name="count")

    result = minority_species(counts)

    assert result == {"fox", "badger"}


def test_minority_species_empty_when_all_equal():
    counts = pd.Series({"bobcat": 10, "coyote": 10}, name="count")

    result = minority_species(counts)

    assert result == set()


def test_build_sample_weights_inverse_frequency():
    counts = pd.Series({"bobcat": 100, "fox": 10})
    labels = pd.Series(["bobcat", "bobcat", "fox"])

    weights = build_sample_weights(labels, counts)

    assert weights[0] == pytest.approx(1.0 / 100)
    assert weights[2] == pytest.approx(1.0 / 10)
    assert weights[2] > weights[0]
