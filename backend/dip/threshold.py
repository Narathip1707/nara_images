"""Chapter 6 — Thresholding & segmentation.

Slides: Thresholding_Chapter / Otsu_Chapter / Intermean_Chapter /
        Intermean_Repeat_Chapter / Intermean_adaptation
Plus `Boss/week7 real/`: Local_splits_method (split4, MergeLocal) and
Otsu_adaptation (otsu with an [st, en) window) — these are NOT in `slide Aj`.
"""
import math

import numpy as np

from .common import Steps, calc_hist

EPS = 0.00000001  # the professor's signature guard against divide-by-zero


def float2int(x):
    """Round-half-up, as written in Otsu_adaptation.ipynb (not Python's banker's rounding)."""
    return math.ceil(x) if x - math.floor(x) >= 0.5 else math.floor(x)


def _class_stats(prob, t, st, en):
    """w0, w1, u0, u1 for the split at t over the window [st, en)."""
    w0 = float(np.sum(prob[st:t])) + EPS
    w1 = float(np.sum(prob[t:en])) + EPS
    idx = np.arange(len(prob))
    u0 = float(np.sum(idx[st:t] * prob[st:t])) / w0
    u1 = float(np.sum(idx[t:en] * prob[t:en])) / w1
    return w0, w1, u0, u1


def otsu_value(hist, st=0, en=None):
    """T = argmax_t w0(t)*w1(t)*(u0(t)-u1(t))^2 over t in [st+1, en)."""
    hist = np.asarray(hist, dtype=np.float64).ravel()
    en = len(hist) if en is None else en
    prob = np.zeros_like(hist)
    total = float(np.sum(hist[st:en]))
    if total == 0:
        return st, []
    prob[st:en] = hist[st:en] / total

    trace, coef_max, thr = [], -1.0, -1
    for t in range(st + 1, en):
        w0, w1, u0, u1 = _class_stats(prob, t, st, en)
        coef = (w0 * w1) * ((u0 - u1) ** 2)
        trace.append((t, w0, w1, u0, u1, coef))
        if coef > coef_max:
            coef_max, thr = coef, t
    return thr, trace


def otsu(img, st=0, en=256):
    hist = calc_hist(img)
    thr, trace = otsu_value(hist, st, en)
    out = np.zeros_like(img, dtype=np.uint8)
    out[img >= thr] = 255

    step = Steps("Otsu's Method",
                 "σ²_B(t) = w0(t) · w1(t) · ( μ0(t) − μ1(t) )²\n"
                 "T = argmax σ²_B(t)          สำหรับ t = 1 … 254")
    step.value("T ที่หาได้", thr)
    step.text("หาค่า t ที่ทำให้ความแปรปรวนระหว่างคลาส (inter-class variance) สูงสุด "
              "ซึ่งเทียบเท่ากับทำให้ความแปรปรวนภายในคลาสต่ำสุด เพราะ σ²_T = σ²_W + σ²_B คงที่")
    step.text("ลูป t เริ่มที่ %d ไม่ใช่ 0 และจบที่ %d ไม่ใช่ %d — ถ้าเริ่ม/จบผิดจะหารด้วยศูนย์"
              % (st + 1, en - 1, en))
    step.value("ตัวกันหารศูนย์", "+0.00000001 ทั้ง w0 และ w1")
    if trace:
        step.chart("σ²_B(t) ทุกค่า t", {"σ²_B": [r[5] for r in trace]},
                   marker={"index": thr - (st + 1), "label": "T = %d" % thr},
                   note="จุดสูงสุดของกราฟนี้คือค่า T")
        step.table("ตารางคำนวณทีละ t",
                   ["t", "w0", "w1", "μ0", "μ1", "σ²_B"],
                   [[r[0], round(r[1], 5), round(r[2], 5), round(r[3], 4),
                     round(r[4], 4), round(r[5], 5)] for r in trace],
                   highlight=[thr - (st + 1)])
    return out, step


def intermean(img, t0=None):
    """T_{n+1} = (u0 + u1)/2, iterate until |T_{n+1} - T_n| <= 1."""
    hist = calc_hist(img).astype(np.float64)
    st, en = 0, 256
    prob = hist / np.sum(hist)
    t = int(np.mean(img)) if t0 is None else int(t0)

    rounds, seen = [], set()
    while True:
        w0, w1, u0, u1 = _class_stats(prob, t, st, en)
        t1 = t if (u0 == 0.0 and u1 == 0.0) else float2int((u0 + u1) / 2)
        rounds.append((len(rounds) + 1, t, w0, w1, u0, u1, t1))
        if abs(t1 - t) <= 1 or t1 in seen:
            t = t1
            break
        seen.add(t)
        t = t1

    out = np.zeros_like(img, dtype=np.uint8)
    out[img >= t] = 255

    step = Steps("Intermean (Iterative) Method",
                 "T(n+1) = round( ( μ0 + μ1 ) / 2 )          หยุดเมื่อ  | T(n+1) − T(n) | ≤ 1")
    step.value("T₀ (ค่าเริ่มต้น)", rounds[0][1],
               "ค่าตั้งต้นมาตรฐานคือ int(np.mean(img)) หรือ 128")
    step.value("T สุดท้าย (ลู่เข้าแล้ว)", t)
    step.value("จำนวนรอบที่ใช้", len(rounds))
    step.table("การลู่เข้าทีละรอบ",
               ["รอบ", "T_n", "w0", "w1", "μ0", "μ1", "T_{n+1}"],
               [[r[0], r[1], round(r[2], 5), round(r[3], 5), round(r[4], 4),
                 round(r[5], 4), r[6]] for r in rounds],
               highlight=[len(rounds) - 1])
    step.chart("ค่า T แต่ละรอบ", {"T": [r[1] for r in rounds] + [t]})
    return out, step


def intermean_loop(img, times=3):
    """Multi-level: after each convergence, restart the window at st = T+1."""
    hist = calc_hist(img).astype(np.float64)
    st, en = 0, 256
    found = []
    for _ in range(int(times)):
        if st >= en - 1:
            break
        prob = np.zeros_like(hist)
        total = float(np.sum(hist[st:en]))
        if total == 0:
            break
        prob[st:en] = hist[st:en] / total
        t = float2int((st + en) * 0.5)
        for _ in range(100):
            w0, w1, u0, u1 = _class_stats(prob, t, st, en)
            t1 = float2int((u0 + u1) / 2)
            if abs(t1 - t) <= 1:
                t = t1
                break
            t = t1
        found.append((st, en, t))
        st = t + 1

    thr = found[-1][2] if found else 128
    out = np.zeros_like(img, dtype=np.uint8)
    out[img >= thr] = 255

    step = Steps("Intermean Loop (Multi-level)",
                 "หา T ในหน้าต่าง [st, en)  →  ตั้ง st = T + 1  →  ทำซ้ำ")
    step.value("จำนวนรอบ", int(times))
    step.value("T ที่ใช้จริง (รอบสุดท้าย)", thr)
    step.table("T ของแต่ละรอบ", ["รอบ", "st", "en", "T"],
               [[i + 1, f[0], f[1], f[2]] for i, f in enumerate(found)],
               highlight=[len(found) - 1] if found else None)
    step.chart("histogram", {"ความถี่": hist})
    return out, step


def binarize(img, t=100):
    """out = 255 if r >= T else 0."""
    t = int(t)
    out = np.zeros_like(img, dtype=np.uint8)
    out[img >= t] = 255

    step = Steps("Global Thresholding (Binarize)",
                 "Out(y, x) = 255    ถ้า  I(y, x) ≥ T\n"
                 "Out(y, x) =   0    ถ้าไม่ใช่")
    step.value("T", t, "เลือกเอง 0–255 — ลองเทียบกับ T ที่ Otsu/Intermean หาให้อัตโนมัติ")
    white = int(np.count_nonzero(out))
    step.value("พิกเซลขาว (≥ T)", "%d (%.1f%%)" % (white, 100.0 * white / out.size))
    step.value("พิกเซลดำ (< T)", "%d (%.1f%%)" % (out.size - white,
                                                  100.0 * (out.size - white) / out.size))
    step.chart("histogram + เส้น T", {"ความถี่": calc_hist(img)},
               marker={"index": t, "label": "T = %d" % t})
    return out, step


def split4(img):
    """Cut into 4 quadrants at h//2, w//2 (Local_splits_method.ipynb)."""
    h, w = img.shape[:2]
    h_half, w_half = h // 2, w // 2
    return [img[:h_half, :w_half], img[:h_half, w_half:],
            img[h_half:, :w_half], img[h_half:, w_half:]]


def merge_local(parts, out):
    """Reassemble 4 quadrants (Local_splits_method.ipynb)."""
    h, w = out.shape[:2]
    h_half, w_half = h // 2, w // 2
    out[:h_half, :w_half] = parts[0]
    out[:h_half, w_half:] = parts[1]
    out[h_half:, :w_half] = parts[2]
    out[h_half:, w_half:] = parts[3]
    return out


def split_view(img):
    """Show the 4 quadrants with a visible seam — no thresholding yet."""
    parts = split4(img)
    out = np.zeros_like(img, dtype=np.uint8)
    out = merge_local(parts, out)
    h, w = img.shape[:2]
    out[h // 2 - 1:h // 2 + 1, :] = 255
    out[:, w // 2 - 1:w // 2 + 1] = 255

    step = Steps("split4 — แบ่งภาพ 4 ส่วน", "h_half = h // 2 ,    w_half = w // 2")
    step.value("ขนาดภาพเดิม (h × w)", "%d × %d" % (h, w))
    step.table("ขนาดของแต่ละส่วน", ["ส่วน", "สไลซ์", "ขนาด (h × w)"],
               [["img_11 (บนซ้าย)", "img[:h//2, :w//2]", "%d × %d" % parts[0].shape[:2]],
                ["img_12 (บนขวา)", "img[:h//2, w//2:]", "%d × %d" % parts[1].shape[:2]],
                ["img_21 (ล่างซ้าย)", "img[h//2:, :w//2]", "%d × %d" % parts[2].shape[:2]],
                ["img_22 (ล่างขวา)", "img[h//2:, w//2:]", "%d × %d" % parts[3].shape[:2]]])
    step.text("ถ้า h หรือ w เป็นเลขคี่ ส่วนล่าง/ขวาจะใหญ่กว่า 1 พิกเซล เพราะ slice ท้ายเก็บเศษ")
    return out, step


def local_adaptive(img, method="otsu"):
    """split4 -> threshold each quadrant independently -> MergeLocal.

    This is the answer for images with uneven lighting / a shadow across them:
    one global T cannot fit every region.
    """
    parts = split4(img)
    outs, thrs = [], []
    for p in parts:
        hist = calc_hist(p)
        if method == "intermean":
            prob = hist.astype(np.float64) / max(float(np.sum(hist)), 1.0)
            t = int(np.mean(p))
            for _ in range(100):
                w0, w1, u0, u1 = _class_stats(prob, t, 0, 256)
                t1 = float2int((u0 + u1) / 2)
                if abs(t1 - t) <= 1:
                    t = t1
                    break
                t = t1
        else:
            t, _ = otsu_value(hist, 0, 256)
        thrs.append(t)
        q = np.zeros_like(p, dtype=np.uint8)
        q[p >= t] = 255
        outs.append(q)

    out = merge_local(outs, np.zeros_like(img, dtype=np.uint8))

    global_t, _ = otsu_value(calc_hist(img), 0, 256)
    step = Steps("Local Splits Adaptive Thresholding",
                 "split4  →  หา T ของแต่ละส่วนแยกกัน (T₁ T₂ T₃ T₄)  →  MergeLocal")
    step.value("วิธีหา T ของแต่ละส่วน", "Otsu" if method == "otsu" else "Intermean")
    step.value("T แบบ global (ใช้ทั้งภาพค่าเดียว)", global_t, "ใช้เทียบว่าต่างกันแค่ไหน")
    step.table("T ของแต่ละส่วน (หาอิสระจากกัน)",
               ["ส่วน", "T", "ต่างจาก global"],
               [["img_11 (บนซ้าย)", thrs[0], "%+d" % (thrs[0] - global_t)],
                ["img_12 (บนขวา)", thrs[1], "%+d" % (thrs[1] - global_t)],
                ["img_21 (ล่างซ้าย)", thrs[2], "%+d" % (thrs[2] - global_t)],
                ["img_22 (ล่างขวา)", thrs[3], "%+d" % (thrs[3] - global_t)]])
    spread = max(thrs) - min(thrs)
    step.value("ช่วงห่างของ T (max − min)", spread,
               "ยิ่งห่างมาก ยิ่งแปลว่าภาพมีแสงไม่สม่ำเสมอ และยิ่งคุ้มที่จะใช้ local แทน global")
    return out, step
