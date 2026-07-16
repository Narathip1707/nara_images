"""Every worked example in COS3401_Summary, asserted against our implementation.

If these pass, the code in the app produces the same numbers the summary says to
write on the exam. Run:  python -m pytest tests -q
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dip import colorspace, enhance, filtering, histogram, threshold  # noqa: E402
from dip.common import calc_hist  # noqa: E402

IMAGES = os.path.join(os.path.dirname(__file__), "..", "images")


# ---------------------------------------------------------------- ch.6 Otsu
def test_otsu_worked_example():
    """summary 06: hist [1,2,1,0,0,2,3,1] -> T=3, and the full sigma^2_B table."""
    hist = np.array([1, 2, 1, 0, 0, 2, 3, 1], dtype=np.float64)
    thr, trace = threshold.otsu_value(hist, 0, 8)
    assert thr == 3

    got = {t: coef for t, _, _, _, _, coef in trace}
    expected = {1: 1.69, 2: 4.4805, 3: 5.6067, 4: 5.6067, 5: 5.6067, 6: 3.6817, 7: 1.0678}
    for t, want in expected.items():
        assert got[t] == pytest.approx(want, abs=1e-3), f"sigma^2_B({t})"


def test_otsu_picks_first_of_tied_maxima():
    """t=3,4,5 all score 5.6067; strict `coef > coef_max` must keep the first."""
    hist = np.array([1, 2, 1, 0, 0, 2, 3, 1], dtype=np.float64)
    thr, _ = threshold.otsu_value(hist, 0, 8)
    assert thr == 3


# ------------------------------------------------------------ ch.6 Intermean
def test_intermean_worked_example():
    """summary 06: same histogram, T0=3 -> converges immediately at T=3."""
    hist = np.array([1, 2, 1, 0, 0, 2, 3, 1], dtype=np.float64)
    prob = hist / hist.sum()
    w0, w1, u0, u1 = threshold._class_stats(prob, 3, 0, 8)
    assert u0 == pytest.approx(1.0, abs=1e-3)
    assert u1 == pytest.approx(5.8333, abs=1e-3)
    assert threshold.float2int((u0 + u1) / 2) == 3


def test_float2int_rounds_half_up():
    assert threshold.float2int(3.4) == 3
    assert threshold.float2int(3.5) == 4   # round-half-up, not banker's rounding
    assert threshold.float2int(2.5) == 3   # Python's round(2.5) gives 2 -- must not
    assert threshold.float2int(3.6) == 4


# --------------------------------------------------------- ch.3 Equalization
def test_equalization_worked_example_4x4():
    """summary 03: 4x4 L=8 matrix -> cdf [0,3,6,8,10,12,14,16], cdf_min=3, mapping."""
    img = np.array([[1, 1, 2, 2], [1, 2, 3, 3], [4, 4, 5, 5], [6, 6, 7, 7]], dtype=np.uint8)
    hist = calc_hist(img)
    assert list(hist[:8]) == [0, 3, 3, 2, 2, 2, 2, 2]

    cdf = hist.cumsum()
    assert list(cdf[:8]) == [0, 3, 6, 8, 10, 12, 14, 16]

    cdf_masked = np.ma.masked_equal(cdf[:8], 0)
    cdf_min = int(cdf_masked.min())
    assert cdf_min == 3
    assert img.size - cdf_min == 13  # the divisor the summary quotes

    # The summary works in L=8; scale the (L-1) factor accordingly. Bin 0 is empty,
    # so masked_equal keeps it out of the maths and it is filled back in as 0.
    mapped = np.round((cdf_masked - cdf_min) / (16 - cdf_min) * 7)
    mapping = np.ma.filled(mapped, 0).astype(int)
    assert list(mapping) == [0, 0, 2, 3, 4, 5, 6, 7]


def test_equalization_cdf_ends_at_total_pixels():
    """The validity check the summary calls out: the last cdf bin == M*N."""
    img = cv2.imread(os.path.join(IMAGES, "lena.png"), 0)
    cdf = calc_hist(img).cumsum()
    assert int(cdf[-1]) == img.size


# ------------------------------------------------------ ch.5 Mean / Median
def test_mean_and_median_on_salt_block():
    """summary 05: [[10,12,11],[13,255,12],[11,10,12]] -> mean 38, median 12."""
    block = np.array([[10, 12, 11], [13, 255, 12], [11, 10, 12]], dtype=np.uint8)
    assert block.sum() == 346
    assert int(346 / 9) == 38                      # mean: 346/9 = 38.44 -> 38
    assert int(np.sort(block.ravel())[4]) == 12    # median: index floor(9/2) = 4


def test_median_kills_salt_mean_does_not():
    """The whole reason median is the salt & pepper answer."""
    img = np.full((7, 7), 12, dtype=np.uint8)
    img[3, 3] = 255  # one salt pixel
    med, _ = filtering.median_filter(img, 3)
    mean, _ = filtering.mean_filter(img, 3)
    assert med[3, 3] == 12                 # salt removed outright
    assert mean[3, 3] > 12                 # salt smeared into the neighbourhood


def test_median_matches_opencv():
    img = cv2.imread(os.path.join(IMAGES, "lena_salt_512.png"), 0)
    ours, _ = filtering.median_filter(img, 3)
    theirs = cv2.medianBlur(img, 3)
    assert np.array_equal(ours[1:-1, 1:-1], theirs[1:-1, 1:-1])


def test_mean_divides_by_k_squared_not_9():
    """The slide's filterImage hardcodes /9; ours must use /(k*k) for every k."""
    img = np.full((11, 11), 100, dtype=np.uint8)
    for k in (3, 5, 7):
        out, _ = filtering.mean_filter(img, k)
        assert out[5, 5] == 100, f"k={k} — a uniform image must survive a mean filter"


def test_filter_borders_are_untouched():
    """out = img.copy() first: the bd-wide border keeps its original pixels."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (20, 20), dtype=np.uint8)
    for fn in (filtering.mean_filter, filtering.median_filter):
        out, _ = fn(img, 3)
        assert np.array_equal(out[0, :], img[0, :])
        assert np.array_equal(out[-1, :], img[-1, :])
        assert np.array_equal(out[:, 0], img[:, 0])


# ------------------------------------------------------ ch.5 Prewitt / Sobel
def test_prewitt_and_sobel_worked_example():
    """summary 05: [[10,12,30],[12,15,35],[11,13,32]] -> Prewitt 64.12, Sobel 87.14."""
    block = np.array([[10, 12, 30], [12, 15, 35], [11, 13, 32]], dtype=np.float64)

    for k, want_gx, want_gy, want_g in ((1, 64, 4, 64.12), (2, 87, 5, 87.14)):
        mx = np.array([[-1, 0, 1], [-k, 0, k], [-1, 0, 1]], dtype=np.float64)
        my = np.array([[-1, -k, -1], [0, 0, 0], [1, k, 1]], dtype=np.float64)
        gx, gy = float((block * mx).sum()), float((block * my).sum())
        assert gx == pytest.approx(want_gx)
        assert gy == pytest.approx(want_gy)
        assert np.hypot(gx, gy) == pytest.approx(want_g, abs=1e-2)


def test_edge_operator_k1_is_prewitt_k2_is_sobel():
    img = cv2.imread(os.path.join(IMAGES, "cameraman.png"), 0)
    p, sp = filtering.edge_operator(img, 1)
    s, ss = filtering.edge_operator(img, 2)
    assert "Prewitt" in sp.title
    assert "Sobel" in ss.title
    assert not np.array_equal(p, s)


# ----------------------------------------------------------- ch.2 Gamma / Log
def test_gamma_worked_example():
    """summary 02: gamma=0.5, r_max=250 -> r=100 gives 161, r=200 gives 228."""
    img = np.array([[100, 200, 250]], dtype=np.uint8)
    out, _ = enhance.gamma(img, 0.5)
    assert int(out[0, 0]) == 161
    assert int(out[0, 1]) == 228
    assert int(out[0, 2]) == 255  # r_max always maps to c = 255


def test_gamma_direction():
    img = cv2.imread(os.path.join(IMAGES, "lena.png"), 0)
    brighter, _ = enhance.gamma(img, 0.4)
    darker, _ = enhance.gamma(img, 2.5)
    assert brighter.mean() > img.mean()
    assert darker.mean() < img.mean()


def test_gamma_1_is_identity_when_rmax_is_255():
    img = np.array([[0, 50, 128, 200, 255]], dtype=np.uint8)
    out, _ = enhance.gamma(img, 1.0)
    assert np.array_equal(out, img)


def test_log_constant_c():
    """summary 02: r_max=250 -> c = 255/ln(251) ~= 46.15."""
    c = 255.0 / np.log(1 + 250.0)
    assert c == pytest.approx(46.15, abs=1e-2)


def test_log_maps_rmax_to_the_top_of_the_range():
    img = np.array([[0, 10, 100, 250]], dtype=np.uint8)
    out, _ = enhance.log_transform(img)
    assert int(out[0, 0]) == 0
    # c is chosen so c*ln(1+r_max) == 255.0 exactly, but .astype(uint8) truncates
    # (and float error nudges it to 254.9999...), so the top pixel lands on 254.
    assert int(out[0, 3]) == 254


def test_log_uses_natural_log_not_log10():
    """The summary flags this explicitly: np.log is ln, not log10."""
    c = 255.0 / np.log(1 + 250.0)
    assert c * np.log(1 + 10) == pytest.approx(110.67, abs=1e-2)
    assert c * np.log10(1 + 10) != pytest.approx(110.67, abs=1e-2)


# ------------------------------------------------------------ ch.2 Bit-plane
def test_bit_plane_worked_example():
    """summary 02: r=139 = 10001011 -> bits 7,3,1,0 are set; 6,5,4,2 are not."""
    img = np.array([[139]], dtype=np.uint8)
    for b in (7, 3, 1, 0):
        out, _ = enhance.bit_plane(img, b)
        assert int(out[0, 0]) == 255, f"bit {b} of 139 should be 1"
    for b in (6, 5, 4, 2):
        out, _ = enhance.bit_plane(img, b)
        assert int(out[0, 0]) == 0, f"bit {b} of 139 should be 0"


def test_bit_planes_reconstruct_original():
    img = cv2.imread(os.path.join(IMAGES, "cameraman.png"), 0)
    total = np.zeros_like(img, dtype=np.int64)
    for b in range(8):
        plane, _ = enhance.bit_plane(img, b)
        total += (plane // 255).astype(np.int64) * (2 ** b)
    assert np.array_equal(total.astype(np.uint8), img)


# --------------------------------------------------------------- ch.2 Negative
def test_negative_is_its_own_inverse():
    img = cv2.imread(os.path.join(IMAGES, "lena.png"), 0)
    once, _ = enhance.negative(img)
    twice, _ = enhance.negative(once)
    assert np.array_equal(twice, img)


# ---------------------------------------------------------- ch.1 Colour spaces
def test_luma_weights_sum_to_0_9999_not_1():
    """The taught weights are 0.2989/0.5870/0.1140, which sum to 0.9999 -- not 1.0.

    (The NTSC constants they approximate, 0.299/0.587/0.114, do sum to 1.0.)
    The 1-in-10000 shortfall is far below a uint8 step, so it never changes a pixel,
    but the numbers to write on the exam are the professor's, not NTSC's.
    """
    from dip.common import LUMA
    assert LUMA == (0.2989, 0.5870, 0.1140)
    assert sum(LUMA) == pytest.approx(0.9999, abs=1e-6)
    assert sum(LUMA) != pytest.approx(1.0, abs=1e-5)


def test_gray_close_to_opencv():
    img = cv2.imread(os.path.join(IMAGES, "lena_color_256.png"))
    ours, _ = colorspace.to_gray(img)
    theirs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert np.abs(ours.astype(int) - theirs.astype(int)).max() <= 2


def test_rgb_to_cmyk_worked_example():
    """summary 01: RGB(100,150,200) -> CMYK(128, 64, 0, 55).

    C lands on exactly 127.5, so the summary's 128 is a rounded hand-calc while
    .astype(uint8) truncates to 127. See TestSummaryRoundsButCodeTruncates.
    """
    r, g, b = 100 / 255, 150 / 255, 200 / 255
    c, m, y = 1 - r, 1 - g, 1 - b
    k = min(c, m, y)
    c, m, y = (c - k) / (1 - k), (m - k) / (1 - k), (y - k) / (1 - k)
    # The summary's four values are all rounded hand-calcs.
    assert [round(v * 255) for v in (c, m, y, k)] == [128, 64, 0, 55]
    # .astype(uint8) truncates instead: C and M each land 1 lower.
    assert [int(v * 255) for v in (c, m, y, k)] == [127, 63, 0, 55]
    assert c * 255 == pytest.approx(127.5, abs=1e-3)
    assert m * 255 == pytest.approx(63.75, abs=1e-3)


class TestSummaryRoundsButCodeTruncates:
    """The summary's hand-calcs ROUND; every slide ends in .astype(uint8), which
    TRUNCATES. Where a value lands near a .5 boundary the two disagree by 1.

    Both appear on the exam -- hand-calculation questions follow the summary,
    "write a program" questions follow the code -- so the app must show both.
    """

    def test_log_r10_summary_says_111_code_gives_110(self):
        c = 255.0 / np.log(1 + 250.0)
        raw = c * np.log(1 + 10)
        assert raw == pytest.approx(110.663, abs=1e-3)
        assert round(raw) == 111   # summary 02
        assert int(raw) == 110     # np.uint8(...)

    def test_log_r100_summary_says_213_code_gives_212(self):
        c = 255.0 / np.log(1 + 250.0)
        raw = c * np.log(1 + 100)
        assert raw == pytest.approx(212.988, abs=1e-3)
        assert round(raw) == 213   # summary 02
        assert int(raw) == 212

    def test_gamma_happens_to_agree(self):
        """Gamma's worked values sit far from a .5 boundary, so both give 161/228."""
        for r, want in ((100, 161), (200, 228)):
            raw = 255 * (r / 250) ** 0.5
            assert int(raw) == want and round(raw) == want

    def test_our_code_follows_the_professors_truncation(self):
        img = np.array([[10, 100, 250]], dtype=np.uint8)
        out, _ = enhance.log_transform(img)
        assert list(out[0]) == [110, 212, 254]


def test_cmyk_round_trip_is_lossless():
    img = cv2.imread(os.path.join(IMAGES, "lena_color_256.png"))
    back, _ = colorspace.cmyk2rgb(img)
    assert np.abs(back.astype(int) - img.astype(int)).max() <= 1


def test_rgb_to_hsv_worked_example():
    """summary 01: RGB(204,51,153) -> H=320, S=0.75, V=0.8."""
    img = np.array([[[153, 51, 204]]], dtype=np.uint8)  # BGR order
    b, g, r = 153 / 255, 51 / 255, 204 / 255
    cmax, cmin = max(r, g, b), min(r, g, b)
    delta = cmax - cmin
    h = 60.0 * (((r - g) / delta) + 4.0) if cmax == b else None
    h = 60.0 * (((g - b) / delta) % 6.0)  # cmax is r here
    assert h == pytest.approx(320.0, abs=0.5)
    assert delta / cmax == pytest.approx(0.75, abs=1e-3)
    assert cmax == pytest.approx(0.8, abs=1e-3)

    out, st = colorspace.rgb2hsv(img, "h")
    assert int(out[0, 0]) == pytest.approx(320 / 360 * 255, abs=1)


def test_hsv_opencv_ranges():
    """summary 01: OpenCV stores H/2 in [0,179], S and V scaled to [0,255]."""
    img = np.array([[[153, 51, 204]]], dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    assert int(hsv[0, 0, 0]) == pytest.approx(160, abs=1)
    assert int(hsv[0, 0, 1]) == pytest.approx(191, abs=1)
    assert int(hsv[0, 0, 2]) == pytest.approx(204, abs=1)


# -------------------------------------------------- ch.6 real images / split4
def test_otsu_close_to_opencv_on_real_image():
    img = cv2.imread(os.path.join(IMAGES, "cells.tif"), 0)
    ours, _ = threshold.otsu_value(calc_hist(img))
    theirs, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert abs(ours - int(theirs)) <= 2


def test_split4_covers_every_pixel():
    img = cv2.imread(os.path.join(IMAGES, "shade.png"), 0)
    parts = threshold.split4(img)
    assert sum(p.size for p in parts) == img.size
    merged = threshold.merge_local(parts, np.zeros_like(img))
    assert np.array_equal(merged, img)


def test_split4_handles_odd_dimensions():
    img = np.arange(7 * 9, dtype=np.uint8).reshape(7, 9)
    parts = threshold.split4(img)
    assert sum(p.size for p in parts) == img.size
    assert np.array_equal(threshold.merge_local(parts, np.zeros_like(img)), img)


def test_local_adaptive_beats_global_on_shaded_image():
    """shade.png has a gradient across it -- the per-quadrant T values must differ."""
    img = cv2.imread(os.path.join(IMAGES, "shade.png"), 0)
    thrs = [threshold.otsu_value(calc_hist(p))[0] for p in threshold.split4(img)]
    assert max(thrs) - min(thrs) > 5, f"quadrant T values barely differ: {thrs}"


def test_binarize_boundary_is_inclusive():
    """out = 255 if r >= T (not >)."""
    img = np.array([[99, 100, 101]], dtype=np.uint8)
    out, _ = threshold.binarize(img, 100)
    assert list(out[0]) == [0, 255, 255]


# ------------------------------------------------------------- ch.3 histogram
def test_equalization_flattens_cdf():
    img = cv2.imread(os.path.join(IMAGES, "lena.png"), 0)
    out, _ = histogram.equalization(img)
    assert out.std() >= img.std()
    assert int(out.max()) == 255
