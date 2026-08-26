"""Bounding-box expansion and image cropping for classifier input preparation."""

from PIL import Image


def expand_bbox(
    bbox_absolute: tuple[int, int, int, int],
    expansion_fraction: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Grow an absolute-pixel [x, y, w, h] bbox by expansion_fraction on each side, clamped to
    image bounds.

    MegaDetector boxes are tight around the animal; a small margin avoids cutting off ears, tails,
    or limbs at the box edge before the crop is fed to the classifier.
    """
    x, y, w, h = bbox_absolute
    dx = round(w * expansion_fraction)
    dy = round(h * expansion_fraction)

    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(image_width, x + w + dx)
    y2 = min(image_height, y + h + dy)

    return (x1, y1, x2 - x1, y2 - y1)


def crop_to_bbox(image: Image.Image, bbox_absolute: tuple[int, int, int, int]) -> Image.Image:
    """Crop image to an absolute-pixel [x, y, w, h] bbox."""
    x, y, w, h = bbox_absolute
    return image.crop((x, y, x + w, y + h))
