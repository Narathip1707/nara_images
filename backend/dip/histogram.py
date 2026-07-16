"""Chapter 3 — Histogram processing & equalization.

Slides: Histogram_chapter / equalization_Chapter
"""
import numpy as np

from .common import Steps, calc_hist


def equalization(img):
    """s_k = round( (cdf(r_k) - cdf_min) / (M*N - cdf_min) * (L-1) ).

    The slide uses the algebraically equivalent min-max form on the masked cdf:
        cdf_m = (cdf_m - cdf_m.min()) / (cdf_m.max() - cdf_m.min()) * 255
    because cdf.max() == M*N always. No parameters.
    """
    hist = calc_hist(img)
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_min = int(cdf_masked.min())
    total = int(img.size)

    cdf_m = (cdf_masked - cdf_min) / (total - cdf_min) * 255.0
    mapping = np.ma.filled(cdf_m, 0).astype(np.uint8)
    out = mapping[img]

    st = Steps("Histogram Equalization",
               "          cdf(r_k) − cdf_min\n"
               "s_k = round( ───────────────── × (L − 1) )\n"
               "           M · N − cdf_min")
    st.text("ไม่มีพารามิเตอร์ — คำนวณจาก histogram ของภาพเองทั้งหมด")
    st.value("M × N (จำนวนพิกเซลทั้งหมด)", total)
    st.value("L - 1", 255)
    st.value("cdf_min (ค่า cdf ที่ไม่เป็นศูนย์ตัวแรก)", cdf_min,
             "ใช้ np.ma.masked_equal(cdf, 0) เพื่อข้าม bin ที่เป็น 0 — เทคนิคที่อาจารย์เน้น")
    st.value("ตัวหาร (M·N − cdf_min)", total - cdf_min)
    st.chart("histogram ก่อน → หลัง", {"ก่อน": hist, "หลัง": calc_hist(out)})
    st.chart("cdf (ความถี่สะสม)", {"cdf": cdf},
             note="cdf ตัวสุดท้ายต้องเท่ากับ M·N พอดี = %d (ใช้ตรวจว่าคำนวณถูก)" % total)
    rows = [[k, int(hist[k]), int(cdf[k]), int(mapping[k])]
            for k in range(256) if hist[k] > 0]
    st.table("ตาราง mapping (แสดงเฉพาะระดับสีที่มีพิกเซลจริง)",
             ["r_k", "n_k", "cdf(r_k)", "s_k"], rows)
    return out, st


def show_histogram(img):
    """Identity op — exists so the UI can show the manual 256-bin count on its own."""
    hist = calc_hist(img)
    st = Steps("Manual Histogram", "Hist[ I[y, x] ] += 1          (วนลูป y, x ทุกพิกเซล)")
    st.text("นับเองด้วยลูป 2 ชั้น — ข้อสอบชอบบังคับให้เขียนแบบนี้ ห้ามใช้ cv2.calcHist")
    st.value("จำนวนพิกเซลทั้งหมด", int(hist.sum()))
    st.value("ระดับสีที่มีพิกเซลมากที่สุด", int(np.argmax(hist)),
             "มี %d พิกเซล" % int(hist.max()))
    st.value("ค่าต่ำสุด / สูงสุดที่พบในภาพ",
             "%d / %d" % (int(np.min(img)), int(np.max(img))))
    st.chart("histogram", {"ความถี่": hist})
    return img.copy(), st
