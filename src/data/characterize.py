"""Dataset characterization: class distribution, per-site distribution, metadata survey."""
from collections.abc import Iterable

import pandas as pd


def class_distribution(
    annotations: pd.DataFrame,
    categories: pd.DataFrame,
    exclude: Iterable[str] = (),
) -> pd.DataFrame:
    """Per-species annotation counts and imbalance ratio (each count relative to the minority class).

    `exclude` names (e.g. "empty", "car") are dropped before counting, since they are not species.
    """
    merged = annotations.merge(categories, left_on="category_id", right_on="id", suffixes=("", "_cat"))
    merged = merged[~merged["name"].isin(set(exclude))]

    counts = merged.groupby("name").size().to_frame("count").sort_values("count", ascending=False)
    min_count = counts["count"].min()
    counts["imbalance_ratio"] = counts["count"] / min_count

    return counts


def per_site_distribution(
    images: pd.DataFrame,
    annotations: pd.DataFrame,
    categories: pd.DataFrame,
    exclude: Iterable[str] = (),
) -> pd.DataFrame:
    """Species counts per camera location, as a location x species pivot table."""
    merged = annotations.merge(categories, left_on="category_id", right_on="id", suffixes=("", "_cat"))
    merged = merged[~merged["name"].isin(set(exclude))]
    merged = merged.merge(images[["id", "location"]], left_on="image_id", right_on="id", suffixes=("", "_img"))

    pivot = merged.pivot_table(index="location", columns="name", values="id", aggfunc="count", fill_value=0)

    return pivot


def metadata_survey(images: pd.DataFrame, day_start_hour: int = 7, day_end_hour: int = 19) -> dict:
    """Resolution distribution and a day/night split.

    Day/night is a proxy derived from date_captured hour-of-day (day_start_hour <= hour <
    day_end_hour counts as day), not from inspecting actual image pixels -- camera traps switch to
    IR illumination automatically at night, but that switch isn't recorded in this metadata, so
    this is an approximation to be validated against real images before being treated as ground
    truth.
    """
    resolution_counts = images.groupby(["height", "width"]).size().to_dict()

    parsed = pd.to_datetime(images["date_captured"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    has_valid_date = parsed.notna()
    hours = parsed.dt.hour

    is_day = has_valid_date & (hours >= day_start_hour) & (hours < day_end_hour)
    is_night = has_valid_date & ~is_day

    return {
        "resolution_counts": resolution_counts,
        "day_count": int(is_day.sum()),
        "night_count": int(is_night.sum()),
        "unparseable_date_count": int((~has_valid_date).sum()),
    }


def aspect_ratio_survey(images: pd.DataFrame) -> dict:
    """Bucket images into landscape/portrait/square by width/height ratio.

    A near-1.0 ratio (0.95-1.05) counts as square; below that portrait, above that landscape.
    """
    ratio = images["width"] / images["height"]
    category = pd.cut(
        ratio,
        bins=[0, 0.95, 1.05, float("inf")],
        labels=["portrait", "square", "landscape"],
    )
    return category.value_counts().to_dict()


def bbox_area_ratio(bboxes: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Bounding box area as a fraction of full image area, per annotation.

    `bboxes` must have `image_id` and `bbox` ([x, y, width, height], pixel coordinates).
    Small ratios indicate an animal occupying little of the frame -- i.e. far from the camera.
    """
    merged = bboxes.merge(images[["id", "height", "width"]], left_on="image_id", right_on="id", suffixes=("", "_img"))

    bbox_area = merged["bbox"].apply(lambda b: b[2] * b[3])
    image_area = merged["height"] * merged["width"]
    merged["area_ratio"] = bbox_area / image_area

    return merged[["image_id", "area_ratio"]]
