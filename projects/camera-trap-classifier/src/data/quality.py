"""Image quality checks: corruption, blur, effective color channel, near-duplicates."""
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image


def is_corrupted(path: Path) -> bool:
    """True if the file fails PIL's integrity check (truncated/malformed image)."""
    try:
        with Image.open(path) as img:
            img.verify()
        return False
    except Exception:
        return True


def is_opencv_readable(path: Path) -> bool:
    """True if OpenCV can decode the file. PIL's `is_corrupted` check can pass files that OpenCV's
    decoder still rejects, so callers should check this before calling `blur_score`.
    """
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) is not None


def blur_score(path: Path) -> float:
    """Variance of the Laplacian. Lower means blurrier; threshold must be calibrated per dataset."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def is_effectively_grayscale(path: Path, tolerance: float = 2.0) -> bool:
    """True if R, G, and B channels are near-identical on average, even if stored as 3-channel.

    Catches IR night captures saved as RGB JPEGs where every pixel has R ~= G ~= B -- checking
    only PIL's `.mode` would miss these, since they report as "RGB" despite looking grayscale.
    Uses a tolerance on mean absolute channel difference rather than exact equality: JPEG
    compression introduces small per-channel rounding noise (observed as ~0.05-0.06 mean |R-G| on
    real genuinely-grayscale-looking sample images) that exact equality incorrectly rejects.
    """
    arr = np.array(Image.open(path).convert("RGB")).astype(int)
    diff_rg = np.abs(arr[..., 0] - arr[..., 1]).mean()
    diff_gb = np.abs(arr[..., 1] - arr[..., 2]).mean()
    return bool(diff_rg <= tolerance and diff_gb <= tolerance)


def find_near_duplicates(paths: list[Path], threshold: int = 5) -> list[tuple[Path, Path]]:
    """Pairs of images whose perceptual hashes differ by at most `threshold` bits (Hamming distance)."""
    hashes = [(path, imagehash.phash(Image.open(path))) for path in paths]

    duplicates = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            path_a, hash_a = hashes[i]
            path_b, hash_b = hashes[j]
            if hash_a - hash_b <= threshold:
                duplicates.append((path_a, path_b))

    return duplicates
