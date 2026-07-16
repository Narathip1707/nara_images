"""A clone into a folder with Thai in its name must still work.

This is not hypothetical: someone cloned the repo into
    D:\\work\\comsci\\4\\COS3104\\โปรแกรมจากเจ๋ง\\nara_images
and nothing worked. cv2.imread returns None and cv2.imwrite returns False —
without raising — for any non-ASCII path on Windows, so every image silently
failed to load and every upload silently wrote nothing.

Run:  python -m pytest tests -q
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dip.common import imread, imwrite  # noqa: E402

THAI_DIR = "โปรแกรมจากเจ๋ง"
THAI_FILE = "รูปทดสอบ.png"


@pytest.fixture
def thai_dir(tmp_path):
    d = tmp_path / THAI_DIR / "nara_images" / "images"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def sample():
    img = np.zeros((40, 60, 3), np.uint8)
    img[10:30, 15:45] = (30, 200, 90)
    return img


def test_cv2_really_does_fail_on_thai_paths(thai_dir, sample):
    """Pin the OpenCV limitation itself, so the reason for our wrappers stays visible.

    If a future OpenCV fixes this, this test fails and the wrappers can go.
    """
    p = str(thai_dir / THAI_FILE)
    assert cv2.imwrite(p, sample) is False, "cv2.imwrite unexpectedly handled a Thai path"
    assert not os.path.exists(p), "cv2.imwrite claimed failure but wrote the file anyway"


def test_imwrite_then_imread_round_trip(thai_dir, sample):
    p = str(thai_dir / THAI_FILE)
    imwrite(p, sample)
    assert os.path.isfile(p), "imwrite did not create the file"

    back = imread(p)
    assert back is not None, "imread returned None for a Thai path"
    assert back.shape == sample.shape
    assert np.array_equal(back, sample)


def test_imread_grayscale_flag(thai_dir, sample):
    p = str(thai_dir / THAI_FILE)
    imwrite(p, sample)
    gray = imread(p, 0)
    assert gray is not None and gray.ndim == 2


def test_imwrite_raises_instead_of_returning_false(thai_dir, sample):
    """cv2.imwrite fails silently; ours must not."""
    with pytest.raises(OSError):
        imwrite(str(thai_dir / "x.notanextension"), sample)


def test_imwrite_creates_missing_parent(tmp_path, sample):
    """The uploads dir can be deleted while the server runs."""
    p = tmp_path / THAI_DIR / "_uploads" / THAI_FILE
    imwrite(str(p), sample)
    assert p.is_file()


def test_imread_missing_file_returns_none(thai_dir):
    assert imread(str(thai_dir / "ไม่มีไฟล์นี้.png")) is None


def test_imread_non_image_returns_none(thai_dir):
    p = thai_dir / "ไม่ใช่รูป.png"
    p.write_text("this is not an image", encoding="utf-8")
    assert imread(str(p)) is None


def test_imread_empty_file_returns_none(thai_dir):
    p = thai_dir / "ว่าง.png"
    p.write_bytes(b"")
    assert imread(str(p)) is None


@pytest.mark.parametrize("ext", [".png", ".jpg", ".bmp", ".tif"])
def test_every_supported_format_round_trips_on_a_thai_path(thai_dir, sample, ext):
    p = str(thai_dir / ("ทดสอบ" + ext))
    imwrite(p, sample)
    assert imread(p) is not None, f"{ext} failed to round-trip"
