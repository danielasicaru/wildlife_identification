"""Error-segmentation helpers: day/night (pixel-based, not the unreliable hour proxy) and
per-site lookup. Occlusion segmentation is not implemented -- only 20 images have manual
occlusion tags, against 88 test crops from different source images, so expected overlap is
near zero and a segmentation on that basis wouldn't be meaningful."""
from pathlib import Path

import pandas as pd

from src.data.quality import is_effectively_grayscale


def day_night_label(path: Path) -> str:
    return "night" if is_effectively_grayscale(path) else "day"


def build_site_lookup(images_df: pd.DataFrame) -> dict[str, str]:
    return images_df.set_index("file_name")["location"].to_dict()
