"""Chapter 2 — Gray level enhancement (point operations).

Slides: enhancement_negative / enhancement_log_transform /
        enhancement_power_gamma / enhancement_Bit_plane_decomposition
"""
import numpy as np

from .common import Steps, u8


def _samples(r_max, n=6):
    """Demo rows for the step-by-step table, spread over [0, r_max].

    Must be derived from r_max, never fixed: on a dark image (dark2.png has
    r_max=50) a hardcoded r=200 makes r/r_max > 1, the result exceeds 255, and
    np.uint8() raises OverflowError on numpy>=2.
    """
    r_max = int(r_max)
    if r_max <= 0:
        return [0]
    return sorted({int(round(r_max * i / (n - 1))) for i in range(n)})


def negative(img):
    """s = 255 - r. No parameters."""
    out = 255 - img
    st = Steps("Image Negative", "s = 255 - r")
    st.text("ไม่มีพารามิเตอร์ — พลิกค่าความสว่างทุกพิกเซลตรงๆ")
    st.table("ตัวอย่างการแมปค่า", ["r", "s = 255 - r"],
             [[r, 255 - r] for r in (0, 64, 128, 192, 255)])
    return u8(out), st


def log_transform(img):
    """s = c*ln(1+r) with c = 255/ln(1+r_max). c is derived, not a free parameter."""
    r_max = float(np.amax(img))
    if r_max <= 0:
        # all-black: c = 255/ln(1) divides by zero and every output is 0 anyway
        st = Steps("Log Transformation", "s = c · ln(1 + r)")
        st.text("ภาพนี้ดำสนิททั้งรูป (r_max = 0) — log transform ไม่มีอะไรให้ดึงขึ้นมา")
        return img.copy(), st

    c = 255.0 / np.log(1.0 + r_max)
    out = c * np.log(1.0 + img.astype(np.float64))

    st = Steps("Log Transformation",
               "s = c · ln(1 + r)          โดย  c = 255 / ln(1 + r_max)")
    st.value("r_max = np.amax(img)", round(r_max, 2))
    st.value("c", round(c, 4), "c ถูกคำนวณให้เอง ไม่ใช่ค่าที่ปรับได้ — บังคับให้ค่าสูงสุดออกมาพอดี 255")
    st.text("np.log คือ log ฐาน e (ln) ไม่ใช่ฐาน 10 — จุดที่พลาดกันบ่อยในข้อสอบ")
    st.table("ตัวอย่างการแมปค่า", ["r", "ln(1+r)", "c·ln(1+r)", "ปัดเศษ", "→ uint8 (ตัดทิ้ง)"],
             [[r, round(float(np.log(1 + r)), 4), round(float(c * np.log(1 + r)), 3),
               int(round(float(c * np.log(1 + r)))),
               int(np.clip(c * np.log(1 + r), 0, 255))] for r in _samples(r_max)],
             note="2 คอลัมน์สุดท้ายคือจุดที่สรุปกับโค้ดอาจารย์ไม่ตรงกัน — "
                  "สรุปคำนวณมือแบบปัดเศษ แต่ .astype(uint8) ตัดทิ้ง")
    lut = c * np.log(1.0 + np.arange(256))
    st.chart("เส้นโค้ง transfer function (r → s)", {"s": np.clip(lut, 0, 255)},
             note="ยกค่าพิกเซลมืดขึ้นมาก บีบค่าพิกเซลสว่างลง → ใช้กับภาพมืด")
    return u8(out), st


def gamma(img, gamma=0.5):
    """s = c*(r/r_max)^gamma with c = 255. Normalizing r first is the exam's key step."""
    g = float(gamma)
    c = 255.0
    r_max = float(np.amax(img))
    if r_max == 0:
        return img.copy(), Steps("Power-Law (Gamma)", "s = c \\cdot r^{\\gamma}")
    norm = img.astype(np.float64) / r_max
    out = c * np.power(norm, g)

    st = Steps("Power-Law (Gamma) Transformation",
               "s = c · (r / r_max)^γ          โดย  c = 255")
    st.value("gamma (γ)", g)
    st.value("r_max = np.amax(img)", round(r_max, 2))
    st.text("ต้อง normalize r ให้อยู่ [0.0, 1.0] ก่อนยกกำลังเสมอ ไม่งั้นค่าล้นช่วง — "
            "สรุปของคุณเรียกว่า 'หัวใจสำคัญในการสอบ'")
    st.value("γ < 1 → ภาพสว่างขึ้น | γ = 1 → เท่าเดิม | γ > 1 → ภาพมืดลง",
             "สว่างขึ้น" if g < 1 else ("เท่าเดิม" if g == 1 else "มืดลง"))
    st.table("ตัวอย่างการแมปค่า",
             ["r", "r/r_max", "(r/r_max)^γ", "255·(...)", "→ uint8 (ตัดทิ้ง)"],
             [[r, round(r / r_max, 4), round((r / r_max) ** g, 4),
               round(c * (r / r_max) ** g, 3),
               int(np.clip(c * (r / r_max) ** g, 0, 255))] for r in _samples(r_max)])
    lut = c * np.power(np.arange(256) / 255.0, g)
    st.chart("เส้นโค้ง transfer function (r → s)", {"s": np.clip(lut, 0, 255)})
    return u8(out), st


def bit_plane(img, bit=7):
    """s = ((r >> bit) & 1) * 255. bit 0 = LSB (noise), bit 7 = MSB (structure)."""
    b = int(bit)
    plane = np.bitwise_and(np.right_shift(img, b), 1)
    out = (plane * 255).astype(np.uint8)

    st = Steps("Bit-Plane Slicing",
               "s = ( ⌊ r / 2^b ⌋  &  1 ) · 255          เขียนแบบ bitwise:  s = ( (r >> b) & 1 ) · 255")
    st.value("bit (b)", b, "b=0 คือ LSB (นอยส์/ไล่เฉด — ที่ใช้ฝังลายน้ำ), b=7 คือ MSB (โครงสร้างหลักของภาพ)")
    st.text("เทียบเท่ากับ (r >> b) & 1 แล้วคูณ 255 ให้เห็นเป็นภาพขาวดำ")
    demo = 139
    st.table(f"ตัวอย่าง: r = {demo} = {demo:08b}₂ (ตัวอย่างในสรุปของคุณ)",
             ["bit", "r >> bit", "& 1", "× 255"],
             [[i, demo >> i, (demo >> i) & 1, ((demo >> i) & 1) * 255] for i in range(7, -1, -1)],
             highlight=[7 - b])
    st.value("จำนวนพิกเซลที่บิตนี้เป็น 1", int(plane.sum()))
    return out, st
