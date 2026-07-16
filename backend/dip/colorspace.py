"""Chapter 1 — Basic operations & colour spaces.

Slides: Basic_Image_Processing_edited / Gray_level_image_Processing /
        CMYK_ColorSpace_new / HSV_ColorSpace_new

Images come in as BGR (cv2.imread order). Channel index: 0=B, 1=G, 2=R.
"""
import cv2
import numpy as np

from .common import LUMA, Steps, is_color, to_gray as gray_of, u8


def rgb2hsv_full(img):
    """BGR uint8 -> (h, s, v) float64 planes. H in [0,360), S and V in [0,1].

    Lifted verbatim out of rgb2hsv() so every colour function shares one hue.
    Deliberately NOT cv2.cvtColor(BGR2HSV): the professor computes H by hand in
    degrees and never once calls cvtColor for HSV — and cv2 rounds H to an int
    then halves it, so it only ever yields even degrees and disagrees with the
    hand formula at band edges.
    """
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
    return h, s, cmax


def bgr_from_hsv(h, s, v):
    """(h, s, v) -> BGR uint8. The exact inverse of rgb2hsv_full.

    Hand-written rather than cv2.cvtColor(..., COLOR_HSV2BGR) for two reasons:
      1) no for-loop can reproduce cv2's answer — it uses its own sector table
         and cvRound internally, and lands +-1 away on ~27% of pixels. The app
         promises that the loop snippet it displays IS what ran, so the engine
         has to be something a loop can match. This one matches exactly.
      2) this version round-trips losslessly (verified over the whole 8-bit
         cube); the cv2 one is off by 1 on most pixels.

    Rounds (np.rint) instead of truncating, which is the opposite of every other
    function here. color_adjust chains, and with truncation s_factor=v_factor=1
    would not be the identity: every pixel would drop 1 per pass and the image
    would visibly darken each time it was applied.
    """
    c = v * s
    x = c * (1.0 - np.abs(((h / 60.0) % 2.0) - 1.0))
    m = v - c
    i = np.floor(h / 60.0).astype(np.int64) % 6
    z = np.zeros_like(h)
    r, g, b = z.copy(), z.copy(), z.copy()
    for k, (rr, gg, bb) in enumerate([(c, x, z), (x, c, z), (z, c, x),
                                      (z, x, c), (x, z, c), (c, z, x)]):
        sel = i == k
        r[sel], g[sel], b[sel] = rr[sel], gg[sel], bb[sel]
    f = lambda a: np.clip(np.rint((a + m) * 255.0), 0, 255)  # noqa: E731
    return np.dstack([f(b), f(g), f(r)]).astype(np.uint8)


def hue_band(h, s, v, low, high, s_min=0.0, v_min=0.0):
    """Pixels whose hue is in [low, high] AND S >= s_min AND V >= v_min.

    Endpoints inclusive, matching slide Aj's (h >= 0) & (h <= 40).

    Wraps when low > high, which is the only way to express red: it straddles
    0 deg, so (h >= 340) & (h <= 20) is empty but (h >= 340) | (h <= 20) is not.

    The S/V floors reject pixels whose hue is meaningless. Both are needed and
    they catch different things:
      - V floor  -> near-black pixels. S = delta/Cmax explodes when Cmax is tiny,
                    so RGB(3,9,1) reads as "fully saturated green" and lands in
                    the green band. Only V rejects it.
      - S floor  -> gray and blown-out pixels. delta = 0 makes H = 0, so every
                    gray pixel claims to be pure red. V cannot see this (gray 128
                    has V = 0.5); only S can.
    """
    span = float(high) - float(low)
    lo, hi = float(low) % 360.0, float(high) % 360.0
    if span >= 360.0 or span <= -360.0:
        band = np.ones(h.shape, dtype=bool)  # whole circle
    elif lo <= hi:
        band = (h >= lo) & (h <= hi)         # 20..60 -- the ordinary case
    else:
        band = (h >= lo) | (h <= hi)         # 340..20 -- wraps through red
    return band & (s >= float(s_min)) & (v >= float(v_min))


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
    h, s, v = rgb2hsv_full(img)

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
    """Keep only pixels whose hue falls in [low, high] — colour segmentation.

    This is slide Aj HSV_ColorSpace_new cell 12, kept exactly as taught, traps
    and all. Do NOT add wrap-around or an S/V gate here — a fixed version lives
    in color_select(). This one has to stay answerable on the exam, and its two
    failure modes are the lesson:
      - no S/V gate: delta = 0 makes H = 0, so every black and gray pixel reads
        as pure red and a photo on a black background comes back untouched.
      - single interval: red straddles 0 deg, so (h >= 340) & (h <= 20) is empty.
    test_hue_mask_keeps_the_professors_trap_on_purpose pins both.
    """
    if not is_color(img):
        return img.copy(), Steps("Hue Mask").text("ต้องใช้ภาพสี")
    h, _s, _v = rgb2hsv_full(img)
    mask = (h >= float(low)) & (h <= float(high))
    out = np.zeros_like(img)
    out[mask] = img[mask]

    st = Steps("Hue Mask (แยกวัตถุตามเนื้อสี) — ฉบับอาจารย์",
               "mask = (H ≥ low) & (H ≤ high)\n"
               "out[mask] = img[mask]          พิกเซลที่ไม่เข้าเงื่อนไขถูกตั้งเป็น 0 (ดำ)")
    st.value("ช่วง Hue ที่เก็บไว้", "%g° – %g°" % (float(low), float(high)))
    kept = int(np.count_nonzero(mask))
    st.value("พิกเซลที่ผ่าน mask", "%d จาก %d (%.1f%%)" % (kept, mask.size, 100.0 * kept / mask.size))
    st.text("นี่คือโค้ดในสไลด์ cell 12 เป๊ะๆ — คำนวณ H เองด้วยสูตรมือ (องศา 0–360) "
            "อาจารย์ไม่เคยใช้ cv2.cvtColor(BGR2HSV) เลยสักที่ในสไลด์หรือ week1–11")
    st.text("⚠️ กับดักที่ 1 — ไม่มีตัวกรอง S/V: พิกเซลดำ/เทา/ขาวมี Δ=0 → H=0 → ถูกนับเป็น 'สีแดง' "
            "ทั้งหมด ลองกับ lena.png ที่ low=0 high=40 จะเก็บไว้ 45% ส่วน Color Select ที่กรอง S/V "
            "แล้วเก็บ 40% — ส่วนต่างคือพิกเซลซีดที่ไม่ใช่สีแดงจริง")
    st.text("⚠️ กับดักที่ 2 — เลือกสีแดงไม่ได้: แดงคร่อม 0° (340–360 ∪ 0–20) "
            "แต่ (H≥low)&(H≤high) เป็นช่วงเดียว ตั้ง low=340 high=20 จะได้ภาพดำสนิท")
    st.text("→ ทั้ง 2 ข้อแก้แล้วใน 'Color Select' (เมนูถัดไป) แต่ตัวนี้เก็บไว้ตามสไลด์เพื่อใช้ตอบข้อสอบ")
    st.table("ช่วง Hue ของสีหลัก (ใช้อ้างอิงตอนตั้งค่า)",
             ["สี", "ช่วง H โดยประมาณ"],
             [["แดง", "0–20° และ 340–360°"], ["ส้ม/เหลือง", "20–60°"], ["เขียว", "60–170°"],
              ["ฟ้า/น้ำเงิน", "170–260°"], ["ม่วง/ชมพู", "260–340°"]],
             note="ในไฟนอลข้อ 2 คุณใช้ (h>180)&(h<260) จับสีฟ้า และ (h>20)&(h<60) จับสีส้ม")
    return out, st


_PRESET_NAMES = [("แดง", 340, 20), ("ส้ม/เหลือง", 20, 60), ("เขียว", 60, 170),
                 ("ฟ้า", 170, 200), ("น้ำเงิน", 200, 260), ("ม่วง/ชมพู", 260, 340)]


def _clean_bands(bands, max_bands=6):
    """Validate the `bands` param. /api/apply does no type checking at all, so a
    malformed value would reach numpy and surface as an opaque 500."""
    if bands is None:
        return [[20.0, 60.0]]
    if not isinstance(bands, (list, tuple)) or not len(bands):
        raise ValueError("ต้องเลือกช่วง Hue อย่างน้อย 1 ช่วง")
    out = []
    for band in bands[:max_bands]:
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise ValueError("แต่ละช่วง Hue ต้องเป็นคู่ [low, high] เช่น [20, 60]")
        try:
            lo, hi = float(band[0]), float(band[1])
        except (TypeError, ValueError):
            raise ValueError("ช่วง Hue ต้องเป็นตัวเลข (องศา 0–360)")
        out.append([lo, hi])
    return out


def _bands_mask(h, s, v, bands, s_min, v_min):
    mask = np.zeros(h.shape, dtype=bool)
    for lo, hi in bands:
        mask |= hue_band(h, s, v, lo, hi, s_min, v_min)
    return mask


def _band_label(lo, hi):
    for name, plo, phi in _PRESET_NAMES:
        if abs(lo - plo) < 1e-9 and abs(hi - phi) < 1e-9:
            return "%s (%g–%g°)" % (name, lo, hi)
    return "%g–%g°" % (lo, hi)


def color_select(img, bands=None, s_min=0.25, v_min=0.15, drop="gray"):
    """Keep the selected hue bands; everything else is dropped to gray/black/white.

    The fixed version of hue_mask: wrap-aware, S/V-gated, and multi-band. drop="black"
    is the professor's cell 12 and your final's focus_mask_hsv; drop="gray" is the
    colour-pop look (your final does it with saturation_adjust(s, 0.2, mask)).
    """
    if not is_color(img):
        return img.copy(), Steps("Color Select").text("ต้องใช้ภาพสี")
    bands = _clean_bands(bands)
    h, s, v = rgb2hsv_full(img)
    mask = _bands_mask(h, s, v, bands, s_min, v_min)

    if drop == "keep":
        out = np.dstack([mask.astype(np.uint8) * 255] * 3)
    else:
        if drop == "black":
            base = np.zeros_like(img)
        elif drop == "white":
            base = np.full_like(img, 255)
        else:  # gray -- colour pop
            base = np.dstack([gray_of(img)] * 3)
        out = base
        out[mask] = img[mask]

    st = Steps("Color Select (เลือกสี — ที่เหลือดรอป)",
               "mask = OR ของทุกช่วง:  (H ≥ low) & (H ≤ high)   ถ้า low ≤ high\n"
               "                        (H ≥ low) | (H ≤ high)   ถ้า low > high  ← วนผ่าน 0° (สีแดง)\n"
               "        แล้ว AND ด้วย  (S ≥ s_min) & (V ≥ v_min)\n"
               "out = พื้น(เทา/ดำ/ขาว) ;  out[mask] = img[mask]")
    st.table("ช่วง Hue ที่เลือก", ["#", "ช่วง", "วนผ่าน 0°?", "พิกเซลที่ผ่าน"],
             [[i + 1, _band_label(lo, hi), "ใช่" if lo > hi else "ไม่",
               int(np.count_nonzero(hue_band(h, s, v, lo, hi, s_min, v_min)))]
              for i, (lo, hi) in enumerate(bands)])
    kept = int(np.count_nonzero(mask))
    st.value("รวมพิกเซลที่เก็บไว้", "%d จาก %d (%.1f%%)" % (kept, mask.size, 100.0 * kept / mask.size))
    st.value("สิ่งที่ทำกับสีที่ไม่ได้เลือก",
             {"gray": "แปลงเป็นขาวดำ (color pop)", "black": "ดำ (แบบสไลด์อาจารย์)",
              "white": "ขาว", "keep": "แสดง mask อย่างเดียว"}.get(drop, drop))
    st.value("S ต่ำสุด", s_min, "กันพิกเซลเทา/ขาวโพลน — Δ=0 → H=0 → ถูกนับเป็นสีแดง")
    st.value("V ต่ำสุด", v_min, "กันพื้นหลังดำ/นอยส์ — S = Δ/Cmax ระเบิดเมื่อ Cmax เล็ก")
    st.text("ทำไมต้องมีทั้ง S และ V: วัดจริงกับนอยส์เกือบดำ (0–13) พบว่า 13% ตกในช่วงสีส้ม "
            "และมี S เฉลี่ยถึง 0.69 — S floor กรองออกได้แค่ 6% แต่ V floor กรองออก 100% "
            "กลับกัน พิกเซลเทา 128 มี V=0.5 (V กรองไม่ได้) แต่ S=0 (S กรองได้)")
    st.text("ลองเห็นผลชัดๆ: ตั้ง 'ที่เหลือ → ขาว' + เลือกสีแดง (340–20) แล้วลาก S/V ต่ำสุด ลง 0 "
            "จะเห็นพื้นหลังดำโผล่เป็นจุดดำเต็มพื้นขาว เพราะพิกเซลดำมี H=0 เลยถูกนับเป็นสีแดงไปด้วย "
            "(ถ้าเลือก 'ที่เหลือ → เทา' จะไม่เห็นความต่าง เพราะพิกเซลดำที่เก็บไว้กับที่แปลงเป็นเทา "
            "ก็ดำเหมือนกันอยู่ดี — แต่พอเอาไปต่อ Color Adjust ที่ V× > 1 นอยส์จะถูกขยายขึ้นมาให้เห็น)")
    return out, st


def color_adjust(img, bands=None, s_min=0.25, v_min=0.15,
                 s_factor=1.0, v_factor=1.0, h_shift=0.0):
    """Adjust S / V / H only inside the selected bands, then convert back to BGR.

    This is saturation_adjust + value_adjust + hsv2bgr from your final's
    image_tool.py, plus a hue shift.
    """
    if not is_color(img):
        return img.copy(), Steps("Color Adjust").text("ต้องใช้ภาพสี")
    bands = _clean_bands(bands)
    h, s, v = rgb2hsv_full(img)
    mask = _bands_mask(h, s, v, bands, s_min, v_min)

    h, s, v = h.copy(), s.copy(), v.copy()
    s[mask] = np.clip(s[mask] * float(s_factor), 0.0, 1.0)
    v[mask] = np.clip(v[mask] * float(v_factor), 0.0, 1.0)
    h[mask] = (h[mask] + float(h_shift)) % 360.0
    out = bgr_from_hsv(h, s, v)

    st = Steps("Color Adjust (ปรับเฉพาะสีที่เลือก)",
               "mask = ช่วง Hue ที่เลือก (AND กับ S ≥ s_min, V ≥ v_min)\n"
               "s[mask] = clip( s[mask] × s_factor , 0, 1 )\n"
               "v[mask] = clip( v[mask] × v_factor , 0, 1 )\n"
               "h[mask] = ( h[mask] + h_shift ) mod 360\n"
               "out = HSV → BGR")
    st.table("ช่วง Hue ที่ปรับ", ["#", "ช่วง"],
             [[i + 1, _band_label(lo, hi)] for i, (lo, hi) in enumerate(bands)])
    touched = int(np.count_nonzero(mask))
    st.value("พิกเซลที่ถูกปรับ", "%d จาก %d (%.1f%%)" % (touched, mask.size,
                                                        100.0 * touched / mask.size))
    st.value("S × (ความสด)", s_factor,
             "<1 = ซีดลง, >1 = สดขึ้น — ในไฟนอลคุณใช้ 0.2 กับสีฟ้า และ 40.0 กับสีส้ม")
    st.value("V × (ความสว่าง)", v_factor, "ในไฟนอลคุณใช้ 1.3 กับสีส้ม")
    st.value("H เลื่อน (°)", h_shift, "เลื่อนเนื้อสีไปเป็นสีอื่น เช่น +120° เขียว → น้ำเงิน")
    if float(s_factor) > 5.0:
        st.text("หมายเหตุ: S ถูก clip ที่ 1.0 อยู่แล้ว ค่า 40 กับ 5 จึงให้ผลเหมือนกัน — "
                "ในไฟนอลคุณใส่ 40.0 ซึ่งก็แค่ดันให้สดเต็มเพดาน")
    st.text("⚠️ ตรงนี้ที่เดียวในแอปที่ 'ปัดเศษ' (np.rint) แทนที่จะตัดทิ้ง เพราะฟังก์ชันนี้ต่อ pipeline ได้ "
            "ถ้าตัดทิ้ง s_factor=1, v_factor=1 จะไม่ใช่ identity — ทุกพิกเซลลด 1 แล้วภาพจะมืดลงทุกครั้งที่กด")
    return out, st


def hsv2bgr(img):
    """BGR -> HSV -> BGR. Proves the inverse formula, the way cmyk2rgb does."""
    if not is_color(img):
        return img.copy(), Steps("HSV → BGR").text("ต้องใช้ภาพสี")
    h, s, v = rgb2hsv_full(img)
    out = bgr_from_hsv(h, s, v)

    st = Steps("HSV → BGR (แปลงกลับ)",
               "C = V × S\n"
               "X = C × ( 1 − | (H / 60) mod 2 − 1 | )\n"
               "m = V − C\n"
               "(R',G',B') = (C,X,0) (X,C,0) (0,C,X) (0,X,C) (X,0,C) (C,0,X)   ตาม H/60 = 0..5\n"
               "R = (R' + m) × 255 ,  G = (G' + m) × 255 ,  B = (B' + m) × 255")
    diff = int(np.abs(out.astype(int) - img.astype(int)).max())
    st.value("ค่าต่างสูงสุดจากภาพเดิม", diff,
             "0 พอดี — สูตรนี้แปลงไป-กลับได้ไม่เสียข้อมูลเลย (ทดสอบครบ 2 ล้านสีแล้ว)"
             if diff == 0 else "⚠️ ควรเป็น 0")
    st.text("อาจารย์ไม่ได้สอนทางกลับ — สไลด์แปลง RGB→HSV แล้วจบ ไม่เคยแปลงกลับ "
            "แต่ถ้าจะปรับ S/V แล้วแสดงเป็นภาพสี ต้องมีตัวนี้")
    st.text("⚠️ cv2 มีทางลัด cv2.cvtColor(hsv.astype('float32'), COLOR_HSV2BGR) "
            "(โหมด float32 รับ H เป็น 0–360, S/V เป็น 0–1 ตรงกับสูตรอาจารย์พอดี) "
            "แต่มันเพี้ยนจากสูตรมือ ±1 บน ~27% ของพิกเซล และ for-loop เขียนตามให้ตรงเป๊ะไม่ได้ "
            "แอปนี้เลยคำนวณเองเพื่อให้โค้ดที่โชว์ = โค้ดที่รัน")
    return out, st
