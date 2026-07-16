"""Chapter 1 — Basic operations & colour spaces.

Slides: Basic_Image_Processing_edited / Gray_level_image_Processing /
        CMYK_ColorSpace_new / HSV_ColorSpace_new

Images come in as BGR (cv2.imread order). Channel index: 0=B, 1=G, 2=R.
"""
import cv2
import numpy as np

from .common import LUMA, Steps, is_color, u8


def to_gray(img):
    """gray = 0.2989*R + 0.5870*G + 0.1140*B — the linear equation, done by hand."""
    if not is_color(img):
        st = Steps("RGB → Grayscale")
        st.text("ภาพนี้เป็น grayscale อยู่แล้ว")
        return img.copy(), st

    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = LUMA[0] * r + LUMA[1] * g + LUMA[2] * b
    out = u8(gray)

    st = Steps("RGB → Grayscale (Linear Equation)",
               "s = 0.2989·R + 0.5870·G + 0.1140·B")
    st.text("น้ำหนักไม่เท่ากันเพราะตาคนไวต่อสีเขียวมากที่สุด (≈59%) รองมาคือแดง (≈30%) "
            "และน้อยสุดคือน้ำเงิน (≈11%)")
    st.value("ผลรวมน้ำหนัก", round(sum(LUMA), 4), "ต้องเท่ากับ 1 พอดี ไม่งั้นความสว่างรวมจะเพี้ยน")
    st.text("⚠️ cv2.imread คืนค่าเป็น BGR ไม่ใช่ RGB — img[y,x,0] คือ Blue, img[y,x,2] คือ Red "
            "สลับพลาดตรงนี้คือข้อผิดที่เจอบ่อยที่สุด")
    yc, xc = img.shape[0] // 2, img.shape[1] // 2
    pb, pg, pr = (int(img[yc, xc, i]) for i in range(3))
    st.table("ตัวอย่างพิกเซลกลางภาพ (y=%d, x=%d)" % (yc, xc),
             ["ช่อง", "ค่า", "น้ำหนัก", "ผลคูณ"],
             [["R (index 2)", pr, LUMA[0], round(LUMA[0] * pr, 3)],
              ["G (index 1)", pg, LUMA[1], round(LUMA[1] * pg, 3)],
              ["B (index 0)", pb, LUMA[2], round(LUMA[2] * pb, 3)],
              ["รวม → gray", "", "", int(out[yc, xc])]])
    return out, st


def bgr2rgb(img):
    """Swap channel 0 and 2 — what cv2.cvtColor(img, COLOR_BGR2RGB) does under the hood."""
    if not is_color(img):
        return img.copy(), Steps("BGR → RGB").text("ภาพ grayscale ไม่มีช่องสีให้สลับ")
    out = img[:, :, ::-1].copy()
    st = Steps("BGR ↔ RGB (สลับช่องสี)",
               "Out[y, x, 0] = I[y, x, 2]        (B ← R)\n"
               "Out[y, x, 1] = I[y, x, 1]        (G ← G  อยู่ที่เดิม)\n"
               "Out[y, x, 2] = I[y, x, 0]        (R ← B)")
    st.text("สลับแค่ช่อง 0 กับ 2 ช่อง 1 (เขียว) อยู่ที่เดิม")
    st.text("ภาพที่แสดงจะดูสีเพี้ยน เพราะเรากำลังตั้งใจสลับให้ผิด — นี่คือสิ่งที่เกิดขึ้น "
            "เวลาลืม cvtColor ก่อนส่งภาพ cv2 ไปให้ matplotlib แสดง")
    return out, st


def channel_split(img):
    """Stack B, G, R vertically — cv2.split + cv2.vconcat as taught."""
    if not is_color(img):
        return img.copy(), Steps("แยกช่องสี").text("ภาพ grayscale มีช่องเดียว")
    b, g, r = cv2.split(img)
    out = cv2.vconcat([b, g, r])
    st = Steps("Channel Split (B / G / R เรียงบนลงล่าง)")
    st.text("เรียงจากบนลงล่าง: Blue → Green → Red (ตามลำดับ index 0, 1, 2 ของ cv2)")
    st.table("ความสว่างเฉลี่ยของแต่ละช่อง", ["ช่อง", "index", "ค่าเฉลี่ย", "min", "max"],
             [["Blue", 0, round(float(b.mean()), 2), int(b.min()), int(b.max())],
              ["Green", 1, round(float(g.mean()), 2), int(g.min()), int(g.max())],
              ["Red", 2, round(float(r.mean()), 2), int(r.min()), int(r.max())]])
    st.text("ช่องไหนสว่างกว่า = ภาพมีสีนั้นเยอะกว่า")
    return out, st


def rgb2cmyk(img, show="all"):
    """C=1-R, M=1-G, Y=1-B, K=min(C,M,Y), then undercolour removal by (1-K)."""
    if not is_color(img):
        return img.copy(), Steps("RGB → CMYK").text("ต้องใช้ภาพสี")
    b, g, r = (img[:, :, i].astype(np.float64) / 255.0 for i in range(3))
    c, m, y = 1.0 - r, 1.0 - g, 1.0 - b
    k = np.minimum(np.minimum(c, m), y)
    den = 1.0 - k
    den[den == 0] = 1.0  # guard: K=1 (pure black) -> C=M=Y=0
    c = (c - k) / den
    m = (m - k) / den
    y = (y - k) / den

    planes = {"c": c, "m": m, "y": y, "k": k}
    if show in planes:
        out = u8(planes[show] * 255.0)
    else:
        top = cv2.hconcat([u8(c * 255), u8(m * 255)])
        bot = cv2.hconcat([u8(y * 255), u8(k * 255)])
        out = cv2.vconcat([top, bot])

    st = Steps("RGB → CMYK",
               "normalize R, G, B → [0, 1] ก่อน\n"
               "C' = 1 − R ,   M' = 1 − G ,   Y' = 1 − B\n"
               "K  = min( C', M', Y' )\n"
               "C = (C' − K) / (1 − K) ,   M = (M' − K) / (1 − K) ,   Y = (Y' − K) / (1 − K)")
    st.value("แสดง", {"all": "ทั้ง 4 ช่อง (C M / Y K)", "c": "Cyan", "m": "Magenta",
                      "y": "Yellow", "k": "Key/Black"}.get(show, show))
    st.text("ต้อง normalize R,G,B เป็น [0,1] ก่อนเสมอ")
    st.text("K คือส่วนดำที่ทั้ง 3 สีมีร่วมกัน — ดึงออกมาแยกเพื่อประหยัดหมึก (undercolour removal) "
            "เพราะพิมพ์ด้วยหมึกดำถูกกว่าผสม C+M+Y")
    st.text("⚠️ ต้องกันหารศูนย์ตอน K=1 (สีดำสนิท) ไม่งั้น (1−K) = 0 → NaN")
    demo_r, demo_g, demo_b = 100, 150, 200
    dc, dm, dy = 1 - demo_r / 255, 1 - demo_g / 255, 1 - demo_b / 255
    dk = min(dc, dm, dy)
    st.table("ตัวอย่างจากสรุปของคุณ: RGB(100, 150, 200)",
             ["ขั้นตอน", "C", "M", "Y", "K"],
             [["normalize → 1−x", round(dc, 4), round(dm, 4), round(dy, 4), "—"],
              ["K = min(C',M',Y')", "—", "—", "—", round(dk, 4)],
              ["หลังหาร (1−K)", round((dc - dk) / (1 - dk), 4), round((dm - dk) / (1 - dk), 4),
               round((dy - dk) / (1 - dk), 4), round(dk, 4)],
              ["× 255 → 8-bit", int((dc - dk) / (1 - dk) * 255), int((dm - dk) / (1 - dk) * 255),
               int((dy - dk) / (1 - dk) * 255), int(dk * 255)]],
             note="ผลที่สรุปคุณเขียนไว้: CMYK (128, 64, 0, 55)")
    return out, st


def cmyk2rgb(img):
    """R = 255(1-C)(1-K) etc. Round-trips a colour image through CMYK and back."""
    if not is_color(img):
        return img.copy(), Steps("CMYK → RGB").text("ต้องใช้ภาพสี")
    b, g, r = (img[:, :, i].astype(np.float64) / 255.0 for i in range(3))
    c, m, y = 1.0 - r, 1.0 - g, 1.0 - b
    k = np.minimum(np.minimum(c, m), y)
    den = 1.0 - k
    den[den == 0] = 1.0
    c, m, y = (c - k) / den, (m - k) / den, (y - k) / den

    r2 = 255.0 * (1 - c) * (1 - k)
    g2 = 255.0 * (1 - m) * (1 - k)
    b2 = 255.0 * (1 - y) * (1 - k)
    out = np.dstack([u8(b2), u8(g2), u8(r2)])

    st = Steps("CMYK → RGB (แปลงไป-กลับ)",
               "R = 255 · (1 − C) · (1 − K)\n"
               "G = 255 · (1 − M) · (1 − K)\n"
               "B = 255 · (1 − Y) · (1 − K)")
    st.text("แปลง RGB → CMYK แล้วแปลงกลับ ควรได้ภาพเดิม ใช้พิสูจน์ว่าสูตรทั้งสองทางถูกต้อง")
    diff = int(np.abs(out.astype(int) - img.astype(int)).max())
    st.value("ค่าต่างสูงสุดจากภาพเดิม", diff,
             "ควรเป็น 0–1 (คลาดจากการปัดเศษ uint8 เท่านั้น)" if diff <= 2 else "⚠️ ต่างมากผิดปกติ")
    return out, st


def rgb2hsv(img, show="h"):
    """Cmax/Cmin/delta form. H in [0,360), S and V in [0,1]."""
    if not is_color(img):
        return img.copy(), Steps("RGB → HSV").text("ต้องใช้ภาพสี")
    b, g, r = (img[:, :, i].astype(np.float64) / 255.0 for i in range(3))
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    nz = delta > 0
    ir = nz & (cmax == r)
    ig = nz & (cmax == g) & ~ir
    ib = nz & (cmax == b) & ~ir & ~ig
    h[ir] = 60.0 * (((g[ir] - b[ir]) / delta[ir]) % 6.0)
    h[ig] = 60.0 * (((b[ig] - r[ig]) / delta[ig]) + 2.0)
    h[ib] = 60.0 * (((r[ib] - g[ib]) / delta[ib]) + 4.0)
    h[h < 0] += 360.0

    s = np.zeros_like(cmax)
    s[cmax > 0] = delta[cmax > 0] / cmax[cmax > 0]
    v = cmax

    if show == "h":
        out = u8(h / 360.0 * 255.0)
    elif show == "s":
        out = u8(s * 255.0)
    else:
        out = u8(v * 255.0)

    st = Steps("RGB → HSV",
               "Cmax = max(R, G, B) ,   Cmin = min(R, G, B) ,   Δ = Cmax − Cmin\n"
               "H = 0                              ถ้า Δ = 0\n"
               "H = 60 · ( ((G − B) / Δ) mod 6 )   ถ้า Cmax = R\n"
               "H = 60 · ( ((B − R) / Δ) + 2 )     ถ้า Cmax = G\n"
               "H = 60 · ( ((R − G) / Δ) + 4 )     ถ้า Cmax = B\n"
               "S = Δ / Cmax   (= 0 ถ้า Cmax = 0) ,   V = Cmax")
    st.value("แสดงช่อง", {"h": "Hue (เนื้อสี)", "s": "Saturation (ความสด)",
                          "v": "Value (ความสว่าง)"}[show])
    st.text("H = 60·(((G−B)/Δ) mod 6) ถ้า Cmax=R | 60·(((B−R)/Δ)+2) ถ้า Cmax=G | "
            "60·(((R−G)/Δ)+4) ถ้า Cmax=B | H=0 ถ้า Δ=0")
    st.text("⚠️ ช่วงค่าต่างกัน 2 มาตรฐาน — สูตรมือ: H ∈ [0,360), S,V ∈ [0,1] "
            "แต่ OpenCV 8-bit: H ∈ [0,179] (หาร 2), S,V ∈ [0,255] (คูณ 255) ข้อสอบถามช่วงบ่อย")
    st.table("ตัวอย่างจากสรุปของคุณ: RGB(204, 51, 153)",
             ["ค่า", "แบบสูตรมือ", "แบบ OpenCV 8-bit"],
             [["H", "320°", "160"], ["S", "0.75", "191"], ["V", "0.8", "204"]])
    st.value("ช่วง H ที่พบในภาพนี้", "%.1f° – %.1f°" % (float(h.min()), float(h.max())))
    return out, st


def hue_mask(img, low=0, high=40):
    """Keep only pixels whose hue falls in [low, high] — colour segmentation."""
    if not is_color(img):
        return img.copy(), Steps("Hue Mask").text("ต้องใช้ภาพสี")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    h = hsv[:, :, 0] * 2.0  # OpenCV stores H/2 -> back to degrees
    mask = (h >= float(low)) & (h <= float(high))
    out = np.zeros_like(img)
    out[mask] = img[mask]

    st = Steps("Hue Mask (แยกวัตถุตามเนื้อสี)",
               "mask = (H ≥ low) & (H ≤ high)\n"
               "out[mask] = img[mask]          พิกเซลที่ไม่เข้าเงื่อนไขถูกตั้งเป็น 0 (ดำ)")
    st.value("ช่วง Hue ที่เก็บไว้", "%g° – %g°" % (float(low), float(high)))
    kept = int(np.count_nonzero(mask))
    st.value("พิกเซลที่ผ่าน mask", "%d จาก %d (%.1f%%)" % (kept, mask.size, 100.0 * kept / mask.size))
    st.text("⚠️ cv2 เก็บ H เป็น [0,179] (หาร 2 มาแล้ว) ต้องคูณ 2 กลับก่อนเทียบกับองศาจริง")
    st.table("ช่วง Hue ของสีหลัก (ใช้อ้างอิงตอนตั้งค่า)",
             ["สี", "ช่วง H โดยประมาณ"],
             [["แดง", "0–20° และ 340–360°"], ["ส้ม/เหลือง", "20–60°"], ["เขียว", "60–170°"],
              ["ฟ้า/น้ำเงิน", "170–260°"], ["ม่วง/ชมพู", "260–340°"]],
             note="ในไฟนอลข้อ 2 คุณใช้ (h>180)&(h<260) จับสีฟ้า และ (h>20)&(h<60) จับสีส้ม")
    return out, st
