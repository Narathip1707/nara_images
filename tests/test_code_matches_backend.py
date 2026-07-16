"""The `codes/*_loop.py` snippets are what the app tells you to write on the exam.
The backend runs a vectorized equivalent instead, for speed. These tests prove the
two agree pixel-for-pixel — otherwise the app would be teaching one thing and
demonstrating another.

Run:  python -m pytest tests -q
"""
import importlib.util
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "backend"))

from dip import colorspace, enhance, filtering, histogram, threshold, transform  # noqa: E402
from dip.common import imread  # noqa: E402

CODES = os.path.join(ROOT, "codes")
IMAGES = os.path.join(ROOT, "images")


def load(name):
    """Import codes/<name>.py as a module."""
    path = os.path.join(CODES, name + ".py")
    spec = importlib.util.spec_from_file_location("snippet_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gray():
    """Small enough that the nested Python loops stay fast."""
    img = imread(os.path.join(IMAGES, "cameraman.png"), 0)
    return cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)


@pytest.fixture(scope="module")
def salt():
    img = imread(os.path.join(IMAGES, "lena_salt_512.png"), 0)
    return cv2.resize(img, (96, 96), interpolation=cv2.INTER_NEAREST)


@pytest.fixture(scope="module")
def color():
    img = imread(os.path.join(IMAGES, "lena_color_256.png"))
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)


def same(a, b, name):
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    assert np.array_equal(a, b), (
        f"{name}: loop snippet and backend disagree — max diff "
        f"{np.abs(a.astype(int) - b.astype(int)).max()}"
    )


# ------------------------------------------------------------------ ch.1
def test_gray_loop_matches(color):
    mine, _ = colorspace.to_gray(color)
    same(load("gray_loop").to_gray(color), mine, "gray")


def test_transpose_loop_matches(gray):
    mine, _ = transform.transpose(gray)
    same(load("transpose_loop").transpose(gray), mine, "transpose")


# ------------------------------------------------------------------ ch.2
def test_negative_loop_matches(gray):
    mine, _ = enhance.negative(gray)
    same(load("negative_loop").negative(gray), mine, "negative")


@pytest.mark.parametrize("g", [0.4, 0.5, 1.0, 2.2])
def test_gamma_loop_matches(gray, g):
    mine, _ = enhance.gamma(gray, g)
    same(load("gamma_loop").gamma_transform(gray, g), mine, f"gamma {g}")


def test_log_loop_matches(gray):
    mine, _ = enhance.log_transform(gray)
    same(load("log_loop").log_transform(gray), mine, "log")


def test_log_loop_survives_a_255_pixel():
    """The snippet used to compute 1 + r on uint8: 1 + 255 wraps to 0, log(0) is
    -inf, and the whole image came out black. cameraman peaks at 245, so the
    fixture alone never caught it — force a 255 in."""
    img = np.array([[0, 100, 200, 255], [255, 12, 3, 250]], dtype=np.uint8)
    out = load("log_loop").log_transform(img)
    assert out.max() > 0, "log snippet returned an all-black image"
    mine, _ = enhance.log_transform(img)
    same(out, mine, "log with a 255 pixel")


@pytest.mark.parametrize("peak", [200, 245, 254, 255])
def test_log_loop_matches_at_every_peak(peak):
    rng = np.random.default_rng(1)
    img = rng.integers(0, peak + 1, (24, 24), dtype=np.uint8)
    img[0, 0] = peak
    out = load("log_loop").log_transform(img)
    mine, _ = enhance.log_transform(img)
    same(out, mine, f"log r_max={peak}")


@pytest.mark.parametrize("peak", [50, 85, 200, 255])
def test_gamma_loop_matches_at_every_peak(peak):
    rng = np.random.default_rng(2)
    img = rng.integers(0, peak + 1, (24, 24), dtype=np.uint8)
    img[0, 0] = peak
    for g in (0.4, 1.0, 2.2):
        out = load("gamma_loop").gamma_transform(img, g)
        mine, _ = enhance.gamma(img, g)
        same(out, mine, f"gamma {g} r_max={peak}")


@pytest.mark.parametrize("bit", range(8))
def test_bitplane_loop_matches(gray, bit):
    mine, _ = enhance.bit_plane(gray, bit)
    same(load("bitplane_loop").bitplane(gray, bit), mine, f"bitplane {bit}")


# ------------------------------------------------------------------ ch.3
def test_histogram_loop_matches(gray):
    from dip.common import calc_hist
    got = np.asarray(load("hist_loop").histogram(gray)).ravel()
    assert np.array_equal(got, calc_hist(gray))


def test_equalize_loop_matches(gray):
    mine, _ = histogram.equalization(gray)
    same(load("equalize_loop").equalize(gray), mine, "equalize")


# ------------------------------------------------------------------ ch.4
@pytest.mark.parametrize("opt", [0, 1, 2])
def test_flip_loop_matches(gray, opt):
    mine, _ = transform.flip(gray, opt)
    same(load("flip_loop").flip(gray, opt), mine, f"flip {opt}")


# ------------------------------------------------------------------ ch.5
@pytest.mark.parametrize("k", [3, 5])
def test_mean_loop_matches(gray, k):
    mine, _ = filtering.mean_filter(gray, k)
    same(load("mean_loop").mean_filter(gray, k), mine, f"mean k={k}")


@pytest.mark.parametrize("k", [3, 5])
def test_median_loop_matches(salt, k):
    mine, _ = filtering.median_filter(salt, k)
    same(load("median_loop").median_filter(salt, k), mine, f"median k={k}")


@pytest.mark.parametrize("k", [1, 2])
def test_edge_loop_matches(gray, k):
    mine, _ = filtering.edge_operator(gray, k)
    same(load("edge_loop").edge(gray, k), mine, f"edge k={k}")


# ------------------------------------------------------------------ ch.6
@pytest.mark.parametrize("t", [50, 100, 200])
def test_binarize_loop_matches(gray, t):
    mine, _ = threshold.binarize(gray, t)
    same(load("binarize_loop").binarize(gray, t), mine, f"binarize T={t}")


def test_otsu_loop_matches(gray):
    from dip.common import calc_hist
    snippet_T = load("otsu_loop").otsu(gray)
    mine_T, _ = threshold.otsu_value(calc_hist(gray))
    assert snippet_T == mine_T, f"Otsu T: snippet {snippet_T} vs backend {mine_T}"


def test_split4_loop_matches(gray):
    snippet = load("split_view_loop")
    for a, b in zip(snippet.split4(gray), threshold.split4(gray)):
        assert np.array_equal(a, b)


def test_local_adaptive_loop_matches():
    img = cv2.resize(imread(os.path.join(IMAGES, "gray3.png"), 0), (96, 96),
                     interpolation=cv2.INTER_AREA)
    snippet = load("local_adaptive_loop")
    mine, _ = threshold.local_adaptive(img, "otsu")
    same(snippet.local_adaptive(img), mine, "local_adaptive")
