"""Chapter 5 — Spatial filtering.

Slides: mean_filtering_chapter / median_filtering_chapter / edge_detection_chapter

Border handling everywhere follows the slides: `out = img.copy()` first, then only
the inner region [bd, n-bd) is written — so the border keeps its original pixels.
"""
import cv2
import numpy as np

from .common import Steps, u8


def _windows(img, k):
    """All k×k neighbourhoods as a (H-2bd, W-2bd, k, k) view — vectorized equivalent
    of the nested pixel loop, giving bit-identical results."""
    return np.lib.stride_tricks.sliding_window_view(img, (k, k))


def mean_filter(img, k=3):
    """s = (1/k²) * sum of the k×k neighbourhood."""
    k = int(k)
    if k % 2 == 0:
        k += 1
    bd = k // 2
    out = img.astype(np.float64).copy()
    win = _windows(img.astype(np.float64), k)
    out[bd:out.shape[0] - bd, bd:out.shape[1] - bd] = win.mean(axis=(2, 3))

    st = Steps("Mean / Average Filter",
               "s = ( 1 / k² ) · Σ r_i          (บวกค่าเพื่อนบ้าน k×k ทั้งหมด แล้วหารด้วย k²)")
    st.value("k (ขนาดเคอร์เนล)", k, "ต้องเป็นเลขคี่เสมอ")
    st.value("ตัวหาร (k²)", k * k)
    st.value("bd = k // 2 (ขอบที่ไม่ประมวลผล)", bd)
    st.matrix("เคอร์เนล", np.ones((k, k)) / (k * k),
              note="ทุกช่องน้ำหนักเท่ากันหมด = 1/k²")
    demo = np.array([[10, 12, 11], [13, 255, 12], [11, 10, 12]])
    st.matrix("ตัวอย่างจากสรุปของคุณ (บล็อกที่มีจุด salt ตรงกลาง)", demo)
    st.value("ผลรวม / 9", "%d / 9 = %.2f → %d" % (demo.sum(), demo.sum() / 9, int(demo.sum() / 9)))
    st.text("ข้อเสีย: จุด noise 255 ตรงกลางไม่หายไป แต่ถูกเกลี่ยไปเปื้อนพิกเซลข้างเคียง "
            "และภาพเบลอลง — เทียบกับ Median ที่ตัดทิ้งได้สนิท")
    return u8(out), st


def median_filter(img, k=3):
    """s = median of the k×k neighbourhood. The right tool for salt & pepper."""
    k = int(k)
    if k % 2 == 0:
        k += 1
    bd = k // 2
    inx = (k * k) // 2
    out = img.copy()
    win = _windows(img, k).reshape(img.shape[0] - 2 * bd, img.shape[1] - 2 * bd, -1)
    out[bd:out.shape[0] - bd, bd:out.shape[1] - bd] = np.sort(win, axis=2)[:, :, inx]

    st = Steps("Median Filter",
               "s = median{ r_i }          (เรียงค่าใน k×k ทั้ง k² ตัว แล้วหยิบตัวที่ index ⌊k²/2⌋)")
    st.value("k (ขนาดเคอร์เนล)", k, "ต้องเป็นเลขคี่เสมอ")
    st.value("ตำแหน่ง median หลังเรียง = ⌊k²/2⌋", inx,
             "เรียงค่าทั้ง %d ตัวแล้วหยิบตัวที่ index %d (นับจาก 0)" % (k * k, inx))
    st.value("bd = k // 2", bd)
    demo = np.array([[10, 12, 11], [13, 255, 12], [11, 10, 12]])
    st.matrix("ตัวอย่างจากสรุปของคุณ (บล็อกที่มีจุด salt ตรงกลาง)", demo)
    srt = np.sort(demo.ravel())
    st.value("เรียงลำดับ", str(list(map(int, srt))))
    st.value("หยิบตัวที่ index 4", int(srt[4]),
             "จุด salt 255 ถูกดันไปท้ายแถวแล้วทิ้งไปเลย — นี่คือเหตุผลที่ Median ชนะ Mean เรื่อง salt & pepper")
    return out, st


def edge_operator(img, k=2):
    """Unified Prewitt/Sobel from edge_detection_chapter:
        gx = [[-1,0,1],[-k,0,k],[-1,0,1]]   gy = [[-1,-k,-1],[0,0,0],[1,k,1]]
        k=1 -> Prewitt, k=2 -> Sobel.  G = sqrt(gx² + gy²)
    """
    k = int(k)
    name = "Prewitt" if k == 1 else "Sobel"
    mx = np.array([[-1, 0, 1], [-k, 0, k], [-1, 0, 1]], dtype=np.float64)
    my = np.array([[-1, -k, -1], [0, 0, 0], [1, k, 1]], dtype=np.float64)

    f = img.astype(np.float64)
    win = _windows(f, 3)
    gx = (win * mx).sum(axis=(2, 3))
    gy = (win * my).sum(axis=(2, 3))
    # edge_operator_meth starts from np.zeros_like -- unlike mean/median, which start
    # from img.copy() -- so the 1px border stays black instead of keeping the original.
    out = np.zeros_like(f)
    out[1:-1, 1:-1] = np.sqrt(gx ** 2 + gy ** 2)

    st = Steps("%s Operator (edge_operator_meth, k=%d)" % (name, k),
               "G = √( Gx² + Gy² )          โดย Gx, Gy = ผลคูณคอนโวลูชันของเคอร์เนลด้านล่างกับภาพ")
    st.value("k", k, "k = 1 → Prewitt | k = 2 → Sobel (ให้น้ำหนักแถวกลาง 2 เท่า ทนนอยส์กว่า)")
    st.matrix("เคอร์เนล Gx (จับขอบแนวตั้ง)", mx)
    st.matrix("เคอร์เนล Gy (จับขอบแนวนอน)", my)
    demo = np.array([[10, 12, 30], [12, 15, 35], [11, 13, 32]], dtype=np.float64)
    dx, dy = float((demo * mx).sum()), float((demo * my).sum())
    st.matrix("ตัวอย่างจากสรุปของคุณ", demo)
    st.value("Gx", round(dx, 2))
    st.value("Gy", round(dy, 2))
    st.value("G = √(Gx² + Gy²)", round(float(np.hypot(dx, dy)), 4))
    st.text("อาจารย์รวม Prewitt กับ Sobel เป็นฟังก์ชันเดียวโดยใช้ k เป็นตัวแปร — "
            "จำสูตรเดียวตอบได้ทั้ง 2 ข้อ")
    return u8(out), st


def canny(img, low=50, high=150):
    out = cv2.Canny(img, int(low), int(high))
    st = Steps("Canny Edge Detection", "cv2.Canny(img, low, high)")
    st.value("low threshold", int(low))
    st.value("high threshold", int(high))
    st.text("อาจารย์สอนแค่การเรียกใช้ cv2.Canny ไม่ได้ลงรายละเอียดอัลกอริทึม — "
            "ค่ามาตรฐานในสไลด์คือ 50 / 150")
    return out, st
