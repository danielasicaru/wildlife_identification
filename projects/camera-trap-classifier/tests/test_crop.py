import numpy as np
import pytest
from PIL import Image

from src.localization.crop import crop_to_bbox, expand_bbox


def test_expand_bbox_grows_box_by_fraction():
    result = expand_bbox((100, 100, 100, 100), expansion_fraction=0.1, image_width=1000, image_height=1000)

    assert result == (90, 90, 120, 120)


def test_expand_bbox_clamps_to_image_bounds():
    result = expand_bbox((0, 0, 50, 50), expansion_fraction=0.5, image_width=1000, image_height=1000)

    assert result[0] == 0
    assert result[1] == 0


def test_expand_bbox_clamps_to_far_edge():
    result = expand_bbox((80, 80, 20, 20), expansion_fraction=1.0, image_width=100, image_height=100)

    x, y, w, h = result
    assert x + w <= 100
    assert y + h <= 100


def test_crop_to_bbox_produces_correct_size():
    arr = np.zeros((200, 300, 3), dtype="uint8")
    img = Image.fromarray(arr)

    cropped = crop_to_bbox(img, (10, 20, 100, 50))

    assert cropped.size == (100, 50)
