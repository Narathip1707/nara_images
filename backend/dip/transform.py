"""Chapter 4 — Geometric & spatial transformations.

Slides: image_fliping_addition(_color) / image_spatial_transformation(_color)

CONVENTION (non-standard, and exactly what the exam uses):
    row-vector times matrix, multiplied from the RIGHT:
        [x', y', 1] = [x, y, 1] . T
    so tx, ty live in ROW 3 (T[2,0], T[2,1]) — not in column 3.
"""
import numpy as np

from .common import Steps, is_color


def flip(img, opt=0):
    """opt 0 = horizontal (mirror left-right), 1 = vertical, 2 = both (180°), 3 = ไม่เปลี่ยน."""
    opt = int(opt)
    names = {0: "แนวนอน (ซ้าย↔ขวา)", 1: "แนวตั้ง (บน↔ล่าง)", 2: "ทั้งสองแกน (หมุน 180°)", 3: "ไม่เปลี่ยน"}
    if opt == 0:
        out = img[:, ::-1].copy()
        formula = "x' = W − 1 − x ,    y' = y            →  out[y, x] = img[y, W − 1 − x]"
    elif opt == 1:
        out = img[::-1, :].copy()
        formula = "x' = x ,    y' = H − 1 − y            →  out[y, x] = img[H − 1 − y, x]"
    elif opt == 2:
        out = img[::-1, ::-1].copy()
        formula = "x' = W − 1 − x ,    y' = H − 1 − y    →  out[y, x] = img[H − 1 − y, W − 1 − x]"
    else:
        out = img.copy()
        formula = "x' = x ,    y' = y            (คัดลอกภาพเดิม)"

    h, w = img.shape[:2]
    st = Steps("Flipping — %s" % names[opt], formula)
    st.value("opt", opt, "0=แนวนอน, 1=แนวตั้ง, 2=ทั้งสองแกน, 3=ไม่เปลี่ยน")
    st.value("ขนาดภาพ H × W", "%d × %d" % (h, w))
    st.text("เขียนด้วยลูป 2 ชั้นเสมอ — อาจารย์ไม่ได้ใช้ cv2.flip ในสไลด์ และข้อสอบออกทั้ง "
            "แบบให้เขียนโค้ดและแบบให้คำนวณพิกัดด้วยมือ")
    st.table("ตัวอย่างการแมปพิกัด", ["(x, y) เดิม", "→ (x', y') ใหม่"],
             [["(0, 0)", _flip_xy(0, 0, w, h, opt)],
              ["(%d, 0)" % (w - 1), _flip_xy(w - 1, 0, w, h, opt)],
              ["(0, %d)" % (h - 1), _flip_xy(0, h - 1, w, h, opt)],
              ["(%d, %d)" % (w - 1, h - 1), _flip_xy(w - 1, h - 1, w, h, opt)]])
    return out, st


def _flip_xy(x, y, w, h, opt):
    if opt == 0:
        return "(%d, %d)" % (w - 1 - x, y)
    if opt == 1:
        return "(%d, %d)" % (x, h - 1 - y)
    if opt == 2:
        return "(%d, %d)" % (w - 1 - x, h - 1 - y)
    return "(%d, %d)" % (x, y)


def transpose(img):
    """out[i, j] = img[j, i] — the output dimensions swap."""
    out = np.swapaxes(img, 0, 1).copy()
    h, w = img.shape[:2]
    st = Steps("Transpose / Diagonal Flip", "x' = y ,    y' = x            →  out[y, x] = img[x, y]")
    st.value("ขนาดเดิม (H × W)", "%d × %d" % (h, w))
    st.value("ขนาดใหม่ (W × H)", "%d × %d" % (w, h), "ขนาดสลับกัน — ต่างจาก flip ที่ขนาดเท่าเดิม")
    return out, st


def init_transformation():
    return np.identity(3, dtype=float)


def matrix_translate(T, tx, ty):
    Ts = np.identity(3, dtype=float)
    Ts[2, 0], Ts[2, 1] = tx, ty
    return np.matmul(Ts, T)


def matrix_scale(T, sx, sy):
    S = np.identity(3, dtype=float)
    S[0, 0], S[1, 1] = sx, sy
    return np.matmul(S, T)


def matrix_rotate(T, theta):
    """Sign convention taken from image_spatial_transformation.ipynb (the gray version):
        R[0,1] = +sin, R[1,0] = -sin
    The color notebook uses the OPPOSITE signs — see the ⚠️ badge in the UI."""
    ang = (float(theta) * np.pi) / 180.0
    R = np.identity(3, dtype=float)
    R[0, 0], R[0, 1] = np.cos(ang), np.sin(ang)
    R[1, 0], R[1, 1] = -np.sin(ang), np.cos(ang)
    return np.matmul(R, T)


def centroid_image(img):
    return int(img.shape[0] / 2), int(img.shape[1] / 2)


def img_transform(img, T):
    """Forward mapping with int() truncation and a bounds check, exactly as taught.

    Forward mapping leaves holes in the output — that is intentional teaching
    material, and the slide patches it afterwards with cv2.medianBlur(out, 3).
    """
    rows, cols = img.shape[:2]
    out = np.zeros_like(img)
    ys, xs = np.mgrid[0:rows, 0:cols]
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=1)
    new = np.matmul(pts, T)
    xn = new[:, 0].astype(int)
    yn = new[:, 1].astype(int)
    ok = (xn >= 0) & (xn < cols) & (yn >= 0) & (yn < rows)
    if is_color(img):
        out[yn[ok], xn[ok], :] = img[ys.ravel()[ok], xs.ravel()[ok], :]
    else:
        out[yn[ok], xn[ok]] = img[ys.ravel()[ok], xs.ravel()[ok]]
    return out, int(np.count_nonzero(~ok))


def translate(img, tx=30, ty=20):
    T = matrix_translate(init_transformation(), float(tx), float(ty))
    out, dropped = img_transform(img, T)
    st = Steps("Translation (เลื่อนภาพ)",
               "                          ⎡  1   0   0 ⎤\n"
               "[x', y', 1] = [x, y, 1] · ⎢  0   1   0 ⎥\n"
               "                          ⎣ tx  ty   1 ⎦")
    st.value("tx, ty", "%g, %g" % (float(tx), float(ty)), "+x ไปขวา, +y ลงล่าง")
    st.matrix("เมทริกซ์ T", T, note="สังเกต tx, ty อยู่ 'แถวที่ 3' ไม่ใช่คอลัมน์ที่ 3 — เพราะใช้ [x,y,1]·T")
    _affine_common(st, img, T, dropped)
    return out, st


def scale(img, sx=1.15, sy=1.15):
    T = matrix_scale(init_transformation(), float(sx), float(sy))
    out, dropped = img_transform(img, T)
    st = Steps("Scaling (ย่อ/ขยาย)",
               "                          ⎡ sx   0   0 ⎤\n"
               "[x', y', 1] = [x, y, 1] · ⎢  0  sy   0 ⎥\n"
               "                          ⎣  0   0   1 ⎦")
    st.value("sx, sy", "%g, %g" % (float(sx), float(sy)), "< 1 ย่อ, > 1 ขยาย")
    st.matrix("เมทริกซ์ T", T)
    _affine_common(st, img, T, dropped)
    return out, st


def rotate(img, theta=-45.0):
    T = matrix_rotate(init_transformation(), float(theta))
    out, dropped = img_transform(img, T)
    st = Steps("Rotation (หมุนรอบจุด 0,0)",
               "                          ⎡  cos θ   sin θ   0 ⎤\n"
               "[x', y', 1] = [x, y, 1] · ⎢ −sin θ   cos θ   0 ⎥        θ เป็นเรเดียน: rad = θ° × π / 180\n"
               "                          ⎣    0       0     1 ⎦")
    st.value("theta (องศา)", float(theta))
    st.value("แปลงเป็นเรเดียน", "rad = θ × π / 180 = %.4f" % ((float(theta) * np.pi) / 180.0))
    st.matrix("เมทริกซ์ T", T)
    st.text("หมุนรอบจุด (0,0) มุมซ้ายบน ไม่ใช่กลางภาพ — ภาพส่วนใหญ่จะหลุดออกนอกกรอบ "
            "ถ้าอยากหมุนรอบกลางภาพต้องใช้ Composite")
    _affine_common(st, img, T, dropped)
    return out, st


def composite(img, theta=-45.0, sx=0.8, sy=0.8):
    """The idiom the slide emphasises: translate to centroid, rotate, scale, translate back."""
    cen = centroid_image(img)
    T = init_transformation()
    T = matrix_translate(T, cen[1], cen[0])
    T = matrix_rotate(T, float(theta))
    T = matrix_scale(T, float(sx), float(sy))
    T = matrix_translate(T, -cen[1], -cen[0])
    out, dropped = img_transform(img, T)

    st = Steps("Composite — หมุน + ย่อ รอบจุดกึ่งกลางภาพ",
               "T = Ts(−xc, −yc) · R(θ) · S(sx, sy) · Ts(xc, yc)\n"
               "    เลื่อนกึ่งกลางไป (0,0)  →  หมุน  →  ย่อ  →  เลื่อนกลับที่เดิม")
    st.value("จุดกึ่งกลาง (yc, xc)", "(%d, %d)" % cen)
    st.value("theta, sx, sy", "%g°, %g, %g" % (float(theta), float(sx), float(sy)))
    st.text("ลำดับสำคัญมาก: เลื่อนกึ่งกลางไปที่ (0,0) ก่อน → หมุน/ย่อ → แล้วเลื่อนกลับ "
            "ถ้าไม่ทำแบบนี้ ภาพจะหมุนรอบมุมซ้ายบนแล้วหลุดกรอบ")
    st.matrix("เมทริกซ์ T รวมสุดท้าย", T)
    _affine_common(st, img, T, dropped)
    return out, st


def _affine_common(st, img, T, dropped):
    rows, cols = img.shape[:2]
    demo = [(4, 2), (0, 0), (cols - 1, rows - 1)]
    st.table("ตัวอย่างการคูณพิกัด [x, y, 1] · T",
             ["(x, y) เดิม", "→ (x', y') ใหม่", "อยู่ในกรอบภาพ?"],
             [[str(p), _apply_pt(p, T), _in_bounds(_apply_pt_raw(p, T), cols, rows)] for p in demo])
    st.value("พิกเซลที่หลุดออกนอกกรอบ (ถูกทิ้ง)",
             "%d จาก %d (%.1f%%)" % (dropped, rows * cols, 100.0 * dropped / (rows * cols)))
    st.text("⚠️ อาจารย์ใช้ forward mapping + int() ตัดเศษ → ผลลัพธ์จะมี 'รูโหว่' เป็นจุดดำ "
            "นี่คือของจริงตามสไลด์ ไม่ใช่บั๊ก และสไลด์แก้ด้วย cv2.medianBlur(out, 3) ทีหลัง")


def _apply_pt_raw(p, T):
    return np.matmul(np.array([p[0], p[1], 1], dtype=float), T)


def _apply_pt(p, T):
    v = _apply_pt_raw(p, T)
    return "(%d, %d)" % (int(v[0]), int(v[1]))


def _in_bounds(v, cols, rows):
    return "อยู่" if (0 <= int(v[0]) < cols and 0 <= int(v[1]) < rows) else "หลุด → ถูกทิ้ง"
