"""Every registered function against every image it recommends.

The gap this closes: the step-by-step tables used to build demo rows from fixed
pixel values (r = 0, 50, 100, 150, 200). On a dark image — dark2.png has
r_max = 50 — r/r_max exceeds 1, the gamma result exceeds 255, and np.uint8()
raises OverflowError on numpy >= 2. numpy 1.26 only warned, so it passed
locally and blew up for anyone who installed numpy fresh. Both dark.jpg and
dark2.png are exactly the images the app recommends for Gamma and Log.

Run:  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "backend"))

from dip.common import imread, is_color, to_gray  # noqa: E402
from registry import FUNCTIONS  # noqa: E402

IMAGES = os.path.join(ROOT, "images")


def _load(name):
    img = imread(os.path.join(IMAGES, name))
    if img is None:
        pytest.skip(f"ไม่มีรูป {name} (รัน get_images.bat ก่อน)")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _cases():
    for f in FUNCTIONS:
        for name in (f.get("good_for") or []):
            yield pytest.param(f, name, id=f"{f['id']}-{name}")


@pytest.mark.parametrize("spec,image_name", list(_cases()))
def test_recommended_image_actually_works(spec, image_name):
    """Every good_for image must survive its own function, defaults and all."""
    img = _load(image_name)

    need = spec.get("input", "any")
    if need == "gray" and is_color(img):
        img = to_gray(img)
    if need == "color" and not is_color(img):
        pytest.fail(f"{spec['id']} ต้องใช้ภาพสี แต่ good_for แนะนำ {image_name} ซึ่งเป็นภาพเทา")

    params = {p["key"]: p["default"] for p in spec.get("params", [])}
    out, step = spec["fn"](img, **params)

    assert out is not None
    assert out.dtype == np.uint8, f"{spec['id']} คืน dtype {out.dtype} ไม่ใช่ uint8"
    if step is not None:
        step.to_dict()  # must be JSON-serialisable for the frontend


DARK = ["dark.jpg", "dark2.png"]


@pytest.mark.parametrize("name", DARK)
@pytest.mark.parametrize("gamma", [0.1, 0.4, 0.5, 1.0, 2.5, 5.0])
def test_gamma_on_dark_images_every_slider_value(name, gamma):
    """dark2.png has r_max=50 — the case that overflowed."""
    from dip import enhance
    img = to_gray(_load(name))
    assert int(np.amax(img)) < 200, f"{name} ไม่มืดพอที่จะเป็นเคสทดสอบนี้"
    out, step = enhance.gamma(img, gamma)
    assert out.dtype == np.uint8
    step.to_dict()


@pytest.mark.parametrize("name", DARK)
def test_log_on_dark_images(name):
    from dip import enhance
    img = to_gray(_load(name))
    out, step = enhance.log_transform(img)
    assert out.dtype == np.uint8
    step.to_dict()


@pytest.mark.parametrize("r_max", [1, 5, 50, 85, 199, 200, 254, 255])
def test_gamma_never_overflows_at_any_r_max(r_max):
    """Directly pin the overflow: sample rows must stay inside [0, 255]."""
    from dip import enhance
    img = np.array([[0, r_max // 2, r_max]], dtype=np.uint8)
    for g in (0.1, 0.5, 1.0, 3.0):
        out, step = enhance.gamma(img, g)
        assert out.max() <= 255
        step.to_dict()


def test_gamma_sample_rows_never_exceed_r_max():
    from dip.enhance import _samples
    for r_max in (0, 1, 7, 50, 85, 255):
        rows = _samples(r_max)
        assert max(rows) <= r_max, f"r_max={r_max} -> {rows} มีค่าเกิน r_max"
        assert min(rows) >= 0


def test_all_gray_functions_survive_a_flat_black_image():
    """r_max = 0 is the degenerate divide-by-zero case."""
    black = np.zeros((32, 32), np.uint8)
    for spec in FUNCTIONS:
        if spec.get("input") == "color":
            continue
        params = {p["key"]: p["default"] for p in spec.get("params", [])}
        out, step = spec["fn"](black, **params)
        assert out is not None, f"{spec['id']} คืน None บนภาพดำล้วน"
        if step is not None:
            step.to_dict()


def test_all_gray_functions_survive_a_flat_white_image():
    white = np.full((32, 32), 255, np.uint8)
    for spec in FUNCTIONS:
        if spec.get("input") == "color":
            continue
        params = {p["key"]: p["default"] for p in spec.get("params", [])}
        out, step = spec["fn"](white, **params)
        assert out is not None, f"{spec['id']} คืน None บนภาพขาวล้วน"
        if step is not None:
            step.to_dict()


@pytest.mark.parametrize("fill", [0, 128, 255])
def test_all_colour_functions_survive_a_flat_image(fill):
    """The two tests above skip input=="color", so the colour path had never been
    run against a flat black image — the exact case that started this feature (a
    photo on a black background). Every one of these is degenerate for HSV:
    delta = 0 everywhere so H is 0, and at fill=0 Cmax = 0 so s = delta/Cmax is 0/0.
    """
    img = np.full((32, 32, 3), fill, np.uint8)
    for spec in FUNCTIONS:
        if spec.get("input") != "color":
            continue
        params = {p["key"]: p["default"] for p in spec.get("params", [])}
        out, step = spec["fn"](img, **params)
        assert out is not None, f"{spec['id']} คืน None บนภาพสีล้วน {fill}"
        assert out.dtype == np.uint8, f"{spec['id']} คืน dtype {out.dtype}"
        if step is not None:
            step.to_dict()


def test_pick_reports_hue_and_flags_weak_pixels():
    """requirements.txt has no httpx, so fastapi.testclient will not import — and
    adding a dependency to an offline app for one test is how the python-multipart
    breakage happened. Test the helper directly instead."""
    import app

    img = np.zeros((4, 4, 3), np.uint8)
    img[0, 0] = (153, 51, 204)  # BGR -> the summary's RGB(204,51,153), H=320
    hit = app._pick_at(img, 0, 0, 20)
    assert hit["h"] == pytest.approx(320.0, abs=0.1)
    assert hit["rgb"] == [204, 51, 153]
    assert hit["band"] == [300.0, 340.0]
    assert hit["weak"] is False
    # the black background: H=0 is meaningless, and saying "you picked red" is worse
    # than saying nothing
    assert app._pick_at(img, 3, 3, 20)["weak"] is True
    # a click past the edge must clamp, not raise
    assert app._pick_at(img, 999, -5, 20)["x"] == 3
