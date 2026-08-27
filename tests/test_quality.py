import numpy as np
import pytest
from PIL import Image, ImageFilter

from src.data import quality
from src.data.quality import (
    blur_score,
    find_near_duplicates,
    is_corrupted,
    is_effectively_grayscale,
    is_opencv_readable,
    is_valid_image,
)


def _checkerboard(path, size=64, square=8):
    arr = np.indices((size, size)).sum(axis=0) % (2 * square) < square
    img = Image.fromarray((arr * 255).astype("uint8")).convert("RGB")
    img.save(path)


def test_is_corrupted_false_for_valid_image(tmp_path):
    path = tmp_path / "valid.jpg"
    _checkerboard(path)

    assert is_corrupted(path) is False


def test_is_corrupted_true_for_garbage_bytes(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not a real jpeg file")

    assert is_corrupted(path) is True


def test_blur_score_higher_for_sharp_image(tmp_path):
    sharp_path = tmp_path / "sharp.jpg"
    blurry_path = tmp_path / "blurry.jpg"

    _checkerboard(sharp_path)
    img = Image.open(sharp_path).filter(ImageFilter.GaussianBlur(radius=8))
    img.save(blurry_path)

    assert blur_score(sharp_path) > blur_score(blurry_path)


def test_is_effectively_grayscale_true_for_equal_channels(tmp_path):
    path = tmp_path / "gray.jpg"
    arr = np.full((32, 32, 3), 128, dtype="uint8")
    Image.fromarray(arr).save(path)

    assert is_effectively_grayscale(path) is True


def test_is_effectively_grayscale_false_for_color_image(tmp_path):
    path = tmp_path / "color.jpg"
    arr = np.zeros((32, 32, 3), dtype="uint8")
    arr[..., 0] = 200  # red channel only
    Image.fromarray(arr).save(path)

    assert is_effectively_grayscale(path) is False


def test_find_near_duplicates_flags_identical_images(tmp_path):
    path_a = tmp_path / "a.jpg"
    path_b = tmp_path / "b.jpg"
    path_c = tmp_path / "c.jpg"

    _checkerboard(path_a)
    _checkerboard(path_b)  # identical content
    arr = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype="uint8")
    Image.fromarray(arr).save(path_c)  # unrelated

    duplicates = find_near_duplicates([path_a, path_b, path_c])

    assert (path_a, path_b) in duplicates or (path_b, path_a) in duplicates
    assert not any(path_c in pair for pair in duplicates)


def test_is_effectively_grayscale_true_within_tolerance(tmp_path):
    # Lossless PNG so the saved pixel values are exact -- proves the tolerance logic itself,
    # independent of JPEG compression behavior (which can round tiny noise away unpredictably,
    # as observed when checking this against real downloaded sample images).
    path = tmp_path / "near_gray.png"
    arr = np.full((32, 32, 3), 128, dtype="uint8")
    arr[..., 0] = 129  # off by 1 from the other two channels, everywhere
    Image.fromarray(arr).save(path)

    assert is_effectively_grayscale(path, tolerance=2) is True


def test_is_effectively_grayscale_false_beyond_tolerance(tmp_path):
    path = tmp_path / "not_gray.png"
    arr = np.full((32, 32, 3), 128, dtype="uint8")
    arr[..., 0] = 140  # off by 12, beyond tolerance
    Image.fromarray(arr).save(path)

    assert is_effectively_grayscale(path, tolerance=2) is False


def test_blur_score_raises_clear_error_when_opencv_cannot_decode(tmp_path, monkeypatch):
    path = tmp_path / "undecodable.jpg"
    _checkerboard(path)

    monkeypatch.setattr(quality.cv2, "imread", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="could not decode"):
        blur_score(path)


def test_is_opencv_readable_false_when_cv2_cannot_decode(tmp_path, monkeypatch):
    path = tmp_path / "undecodable.jpg"
    _checkerboard(path)

    monkeypatch.setattr(quality.cv2, "imread", lambda *args, **kwargs: None)

    assert is_opencv_readable(path) is False


def test_is_opencv_readable_true_for_valid_image(tmp_path):
    path = tmp_path / "valid.jpg"
    _checkerboard(path)

    assert is_opencv_readable(path) is True


def test_is_valid_image_true_for_valid_image(tmp_path):
    path = tmp_path / "valid.jpg"
    _checkerboard(path)

    assert is_valid_image(path) is True


def test_is_valid_image_false_for_corrupted_file(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not a real jpeg file")

    assert is_valid_image(path) is False


def test_is_valid_image_false_when_opencv_cannot_decode_despite_passing_pil(tmp_path, monkeypatch):
    path = tmp_path / "undecodable.jpg"
    _checkerboard(path)

    monkeypatch.setattr(quality.cv2, "imread", lambda *args, **kwargs: None)

    assert is_valid_image(path) is False
