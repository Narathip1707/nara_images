"""The catalogue that drives the whole UI.

The frontend fetches this and builds every button and slider from it — to add a
function you only touch this file plus the matching dip/ module.

Fields
  id       key used by /api/apply
  fn       the callable, returning (image, Steps|None)
  stars    how often the summary says it shows up on the exam (5 = most)
  input    "gray" (auto-converted first) | "color" (needs a colour image) | "any"
  good_for image filenames that best demonstrate it
  why      when to reach for this technique — the exam's "problem -> technique" table
  notes    badges: ("warn"|"info", text) — slide bugs and exam traps
  code     {loop, cv2} basenames in codes/
"""
from dip import colorspace, enhance, filtering, histogram, threshold, transform

CHAPTERS = {
    1: "1 — สีและการแปลงพื้นฐาน",
    2: "2 — ปรับปรุงภาพระดับเทา",
    3: "3 — ฮิสโตแกรม",
    4: "4 — การแปลงเชิงเรขาคณิต",
    5: "5 — การกรองเชิงพื้นที่",
    6: "6 — การแบ่งส่วนภาพ",
}

TRUNC_NOTE = (
    "warn",
    "สรุปของคุณคำนวณมือแบบ 'ปัดเศษ' แต่โค้ดอาจารย์ลงท้ายด้วย .astype(uint8) ซึ่ง 'ตัดทิ้ง' — "
    "ค่าที่ตกใกล้ .5 จะต่างกัน 1 หน่วย ตารางด้านล่างแสดงให้ทั้งสองแบบ "
    "ข้อสอบให้คำนวณมือ = ใช้ปัดเศษ | ข้อสอบให้เขียนโปรแกรม = ได้ค่าแบบตัดทิ้ง",
)

LOOP_NOTE = (
    "info",
    "ข้อสอบสั่ง 'จงเขียนโปรแกรม' โดยไม่ระบุเจาะจง → ต้องเขียนลูป for 2 ชั้นเสมอ "
    "ห้ามเรียก cv2 สำเร็จรูป (Academic Safety Rules ในสรุปของคุณ)",
)

GTE_NOTE = (
    "warn",
    "สไลด์ใช้เครื่องหมายไม่ตรงกันตอนเอา T ไปแบ่งภาพ — slide Aj (Otsu_Chapter, "
    "Intermean_Chapter, Thresholding_Chapter) เขียน out[img >= T] = 255 แต่ไฟล์ใน "
    "week7 real (Otsu_adaptation, Intermean_adaptation) เขียน out[img > T] = 255 "
    "เว็บนี้ยึด >= ตาม slide Aj และตรงกับสรุปของคุณที่เขียนว่า 'r ≥ T' — "
    "ต่างกันแค่พิกเซลที่เท่ากับ T พอดี แต่ในข้อสอบคำนวณมือมีผล",
)

FUNCTIONS = [
    # ------------------------------------------------------------------ ch.1
    {
        "id": "gray", "name": "RGB → Grayscale", "chapter": 1, "stars": 5,
        "fn": colorspace.to_gray, "input": "color", "params": [],
        "formula": "s = 0.2989R + 0.5870G + 0.1140B",
        "why": "ขั้นแรกของเกือบทุกอย่าง — ฟิลเตอร์/threshold/edge ต้องใช้ภาพเทาก่อน",
        "good_for": ["lena_color_256.png", "mandril_color.png", "test_color.png"],
        "notes": [
            ("warn", "cv2.imread คืน BGR ไม่ใช่ RGB — img[y,x,0] คือ Blue, img[y,x,2] คือ Red"),
            ("info", "น้ำหนัก 3 ตัวรวมกันได้ 0.9999 ไม่ใช่ 1.0 พอดี (ค่ามาตรฐาน NTSC คือ "
                     "0.299/0.587/0.114 ซึ่งรวมได้ 1) — ต่างกันน้อยกว่า 1 ระดับสี ไม่มีผลกับภาพ "
                     "แต่ตอนตอบข้อสอบให้เขียนตามอาจารย์"),
            LOOP_NOTE,
        ],
        "code": "gray",
    },
    {
        "id": "bgr2rgb", "name": "BGR ↔ RGB (สลับช่องสี)", "chapter": 1, "stars": 4,
        "fn": colorspace.bgr2rgb, "input": "color", "params": [],
        "formula": "Out[y,x,0] = I[y,x,2], Out[y,x,2] = I[y,x,0]",
        "why": "เข้าใจว่าทำไมภาพ cv2 ที่ส่งเข้า matplotlib ตรงๆ ถึงสีเพี้ยน",
        "good_for": ["lena_color_256.png", "RGB_test.png", "test_color.png"],
        "notes": [LOOP_NOTE], "code": "bgr2rgb",
    },
    {
        "id": "split", "name": "แยกช่องสี B / G / R", "chapter": 1, "stars": 3,
        "fn": colorspace.channel_split, "input": "color", "params": [],
        "formula": "cv2.split(img) → vconcat([b, g, r])",
        "why": "ดูว่าภาพมีสีไหนเยอะ และเห็นว่าแต่ละช่องเป็นภาพเทาแยกกัน",
        "good_for": ["RGB_test.png", "lena_color_256.png", "mandril_color.png"],
        "notes": [], "code": "split",
    },
    {
        "id": "cmyk", "name": "RGB → CMYK", "chapter": 1, "stars": 3,
        "fn": colorspace.rgb2cmyk, "input": "color",
        "params": [{"key": "show", "type": "choice", "default": "all", "label": "ช่องที่แสดง",
                    "options": [["all", "ทั้ง 4 ช่อง"], ["c", "Cyan"], ["m", "Magenta"],
                                ["y", "Yellow"], ["k", "Key (ดำ)"]]}],
        "formula": "K = min(1−R, 1−G, 1−B),  C = (1−R−K)/(1−K)",
        "why": "ระบบสีสำหรับงานพิมพ์ (subtractive) — ตรงข้ามกับ RGB ที่เป็นแสง (additive)",
        "good_for": ["lena_color_256.png", "test_color.png", "RGB_test.png"],
        "notes": [
            ("warn", "ต้องกันหารศูนย์ตอน K=1 (ดำสนิท) ไม่งั้น (1−K)=0 → NaN "
                     "ในไฟนอลข้อ 3 คุณใช้ +1e-10 อาจารย์ใช้ denominator[denominator==0] = 1"),
            TRUNC_NOTE,
        ],
        "code": "cmyk",
    },
    {
        "id": "cmyk2rgb", "name": "CMYK → RGB (แปลงกลับ)", "chapter": 1, "stars": 2,
        "fn": colorspace.cmyk2rgb, "input": "color", "params": [],
        "formula": "R = 255(1−C)(1−K)",
        "why": "พิสูจน์ว่าสูตรไป-กลับถูกต้อง (ควรได้ภาพเดิมเป๊ะ)",
        "good_for": ["lena_color_256.png", "test_color.png"],
        "notes": [], "code": "cmyk2rgb",
    },
    {
        "id": "hsv", "name": "RGB → HSV", "chapter": 1, "stars": 3,
        "fn": colorspace.rgb2hsv, "input": "color",
        "params": [{"key": "show", "type": "choice", "default": "h", "label": "ช่องที่แสดง",
                    "options": [["h", "Hue (เนื้อสี)"], ["s", "Saturation (ความสด)"],
                                ["v", "Value (ความสว่าง)"]]}],
        "formula": "Δ = Cmax − Cmin,  S = Δ/Cmax,  V = Cmax",
        "why": "แยก 'เนื้อสี' ออกจาก 'ความสว่าง' → ใช้เลือกวัตถุตามสีได้ทนแสงกว่า RGB",
        "good_for": ["lena_color_256.png", "test_color.png", "RGB_test.png"],
        "notes": [
            ("warn", "ช่วงค่าต่างกัน 2 มาตรฐาน — สูตรมือ: H∈[0,360), S,V∈[0,1] | "
                     "OpenCV 8-bit: H∈[0,179] (หาร 2), S,V∈[0,255] ข้อสอบถามช่วงบ่อยมาก"),
            ("warn", "สไลด์ HSV_ColorSpace_new เขียน return int(h), int(s), int(v) — "
                     "S กับ V อยู่ [0,1] เลยถูกตัดเหลือ 0 ทั้งหมด ใช้ได้แต่ช่อง H "
                     "เว็บนี้เก็บเป็น float แล้วคูณ 255 ตอนแสดงผลแทน"),
        ],
        "code": "hsv",
    },
    {
        "id": "hue_mask", "name": "Hue Mask (แยกวัตถุตามสี)", "chapter": 1, "stars": 3,
        "fn": colorspace.hue_mask, "input": "color",
        "params": [
            {"key": "low", "type": "int", "min": 0, "max": 360, "step": 5, "default": 20,
             "label": "Hue ต่ำสุด (°)"},
            {"key": "high", "type": "int", "min": 0, "max": 360, "step": 5, "default": 60,
             "label": "Hue สูงสุด (°)"},
        ],
        "formula": "mask = (H ≥ low) & (H ≤ high)",
        "why": "แยกวัตถุตามสี — แบบเดียวกับที่คุณทำในไฟนอลข้อ 2 (จับสีฟ้ากับสีส้ม)",
        "good_for": ["test_color.png", "RGB_test.png", "mandril_color.png"],
        "notes": [("warn", "cv2 เก็บ H เป็น [0,179] ต้องคูณ 2 กลับก่อนเทียบกับองศาจริง")],
        "code": "hue_mask",
    },

    # ------------------------------------------------------------------ ch.2
    {
        "id": "negative", "name": "Image Negative", "chapter": 2, "stars": 3,
        "fn": enhance.negative, "input": "gray", "params": [],
        "formula": "s = 255 − r",
        "why": "ภาพทางการแพทย์ (X-ray, mammogram) — กลับขาวดำแล้วเห็นรายละเอียดในบริเวณสว่างชัดขึ้น",
        "good_for": ["Xray_share.jpg", "breast.bmp", "cameraman.png"],
        "notes": [("info", "ไม่มีพารามิเตอร์ และทำ 2 ครั้งจะได้ภาพเดิม (เป็น inverse ของตัวเอง)"),
                  LOOP_NOTE],
        "code": "negative",
    },
    {
        "id": "log", "name": "Log Transformation", "chapter": 2, "stars": 4,
        "fn": enhance.log_transform, "input": "gray", "params": [],
        "formula": "s = c·ln(1+r),  c = 255/ln(1+r_max)",
        "why": "ภาพมืดมากจนมองไม่เห็นรายละเอียด — ดึงค่าพิกเซลมืดขึ้นมาแรงกว่า Gamma",
        "good_for": ["dark2.png", "dark.jpg", "dark_image.png"],
        "notes": [
            ("info", "c ไม่ใช่ค่าที่ปรับได้ — คำนวณจาก r_max เพื่อบังคับให้ผลสูงสุดพอดี 255"),
            ("warn", "np.log คือ ln (ฐาน e) ไม่ใช่ log ฐาน 10 — จุดที่พลาดกันบ่อย"),
            TRUNC_NOTE,
        ],
        "code": "log",
    },
    {
        "id": "gamma", "name": "Power-Law (Gamma)", "chapter": 2, "stars": 5,
        "fn": enhance.gamma, "input": "gray",
        "params": [{"key": "gamma", "type": "float", "min": 0.1, "max": 5.0, "step": 0.1,
                    "default": 0.5, "label": "γ (gamma)"}],
        "formula": "s = 255·(r/r_max)^γ",
        "why": "γ<1 แก้ภาพมืด | γ>1 แก้ภาพสว่างจ้าขาวโพลน — ปรับได้ละเอียดกว่า Log",
        "good_for": ["dark.jpg", "dark2.png", "bright_image.png", "A3.png"],
        "notes": [
            ("warn", "ต้อง normalize r เป็น [0.0, 1.0] ก่อนยกกำลังเสมอ — "
                     "สรุปของคุณเรียกว่า 'หัวใจสำคัญในการสอบ'"),
            ("info", "γ<1 → สว่างขึ้น | γ=1 → เท่าเดิม | γ>1 → มืดลง"),
            LOOP_NOTE,
        ],
        "code": "gamma",
    },
    {
        "id": "bitplane", "name": "Bit-Plane Slicing", "chapter": 2, "stars": 4,
        "fn": enhance.bit_plane, "input": "gray",
        "params": [{"key": "bit", "type": "int", "min": 0, "max": 7, "step": 1, "default": 7,
                    "label": "bit ที่ต้องการดู"}],
        "formula": "s = ((r >> b) & 1) · 255",
        "why": "ดูว่าข้อมูลภาพจริงอยู่บิตบนๆ ส่วนบิตล่างเกือบเป็นนอยส์ — ใช้ฝังลายน้ำที่ bit 0",
        "good_for": ["cameraman.png", "lena.png", "jetplane.tif"],
        "notes": [("info", "bit 7 = MSB เห็นเป็นรูปชัด | bit 0 = LSB เห็นเป็นนอยส์มั่วๆ "
                           "ลองไล่ 7 → 0 จะเห็นรูปค่อยๆ สลายเป็นนอยส์")],
        "code": "bitplane",
    },

    # ------------------------------------------------------------------ ch.3
    {
        "id": "hist", "name": "ดู Histogram", "chapter": 3, "stars": 4,
        "fn": histogram.show_histogram, "input": "gray", "params": [],
        "formula": "Hist[ I[y,x] ] += 1",
        "why": "อ่านภาพให้ออกก่อนเลือกวิธีแก้ — เกาะซ้าย=ภาพมืด, เกาะกลาง=contrast ต่ำ, 2 ยอด=แยกวัตถุได้",
        "good_for": ["lena.png", "dark.jpg", "cells.tif", "bright_image.png"],
        "notes": [LOOP_NOTE], "code": "hist",
    },
    {
        "id": "equalize", "name": "Histogram Equalization", "chapter": 3, "stars": 5,
        "fn": histogram.equalization, "input": "gray", "params": [],
        "formula": "s_k = round( (cdf(r_k) − cdf_min) / (M·N − cdf_min) · (L−1) )",
        "why": "ภาพมัว contrast ต่ำ สีเทากระจุกตัว — ดึง histogram ให้แผ่เต็มช่วงอัตโนมัติ",
        "good_for": ["A3.png", "lena.png", "cortex.png", "dark2.png"],
        "notes": [
            ("info", "ไม่มีพารามิเตอร์เลย — คำนวณจาก histogram ของภาพเองทั้งหมด"),
            ("info", "cdf ตัวสุดท้ายต้องเท่ากับ M·N พอดี ใช้ตรวจว่าคำนวณถูกในห้องสอบได้"),
            ("warn", "ต้องใช้ np.ma.masked_equal(cdf, 0) ข้าม bin ที่เป็นศูนย์ ไม่งั้นระดับสี "
                     "ที่ไม่มีพิกเซลจะดึง cdf_min ผิด"),
        ],
        "code": "equalize",
    },

    # ------------------------------------------------------------------ ch.4
    {
        "id": "flip", "name": "Flipping (พลิกภาพ)", "chapter": 4, "stars": 5,
        "fn": transform.flip, "input": "any",
        "params": [{"key": "opt", "type": "choice", "default": 0, "label": "แกนที่พลิก",
                    "options": [[0, "แนวนอน (ซ้าย↔ขวา)"], [1, "แนวตั้ง (บน↔ล่าง)"],
                                [2, "ทั้งสองแกน (180°)"], [3, "ไม่เปลี่ยน"]]}],
        "formula": "opt=0: x' = W−1−x  |  opt=1: y' = H−1−y",
        "why": "ออกสอบทั้งแบบให้เขียนโค้ดลูป และแบบให้คำนวณพิกัดด้วยมือ",
        "good_for": ["cameraman.png", "lena_color_256.png", "A3.png"],
        "notes": [
            LOOP_NOTE,
            ("warn", "สไลด์เวอร์ชันสีใช้ convention ปนกัน — opt=0 อ่านแบบ backward map "
                     "(out[i,j] = img[i,c−j−1]) แต่ opt=1,2 เขียนแบบ forward map "
                     "(out[r−i−1,j] = img[i,j]) ผลลัพธ์เหมือนกันแต่วิธีคิดคนละแบบ"),
        ],
        "code": "flip",
    },
    {
        "id": "transpose", "name": "Transpose (พลิกทแยง)", "chapter": 4, "stars": 3,
        "fn": transform.transpose, "input": "any", "params": [],
        "formula": "x' = y,  y' = x",
        "why": "สลับแกน — ขนาดภาพสลับจาก H×W เป็น W×H (ต่างจาก flip ที่ขนาดเท่าเดิม)",
        "good_for": ["cameraman.png", "A3.png"],
        "notes": [("info", "ขนาดภาพเปลี่ยน! ต้องสร้าง out ขนาด (cols, rows) ไม่ใช่ zeros_like(img)")],
        "code": "transpose",
    },
    {
        "id": "translate", "name": "Translation (เลื่อน)", "chapter": 4, "stars": 4,
        "fn": transform.translate, "input": "any",
        "params": [
            {"key": "tx", "type": "int", "min": -200, "max": 200, "step": 5, "default": 30,
             "label": "tx (เลื่อนแนวนอน)"},
            {"key": "ty", "type": "int", "min": -200, "max": 200, "step": 5, "default": 20,
             "label": "ty (เลื่อนแนวตั้ง)"},
        ],
        "formula": "[x', y', 1] = [x, y, 1] · Ts",
        "why": "พื้นฐานของ affine ทั้งหมด — และเป็นชิ้นส่วนของการหมุนรอบกึ่งกลางภาพ",
        "good_for": ["cameraman.png", "A3.png"],
        "notes": [
            ("warn", "อาจารย์ใช้ convention แบบ row-vector คูณจากขวา [x,y,1]·T → tx,ty อยู่ "
                     "'แถวที่ 3' (T[2,0], T[2,1]) ไม่ใช่คอลัมน์ที่ 3 แบบตำราทั่วไป จำให้แม่น"),
            ("info", "ต้องเช็คขอบเขต 0 ≤ xn < cols และ 0 ≤ yn < rows ก่อนเขียนทุกครั้ง"),
        ],
        "code": "translate",
    },
    {
        "id": "scale", "name": "Scaling (ย่อ/ขยาย)", "chapter": 4, "stars": 3,
        "fn": transform.scale, "input": "any",
        "params": [
            {"key": "sx", "type": "float", "min": 0.2, "max": 2.0, "step": 0.05, "default": 1.15,
             "label": "sx"},
            {"key": "sy", "type": "float", "min": 0.2, "max": 2.0, "step": 0.05, "default": 1.15,
             "label": "sy"},
        ],
        "formula": "S(sx, sy) = diag(sx, sy, 1)",
        "why": "ย่อ/ขยายด้วยเมทริกซ์ — ต่างจาก cv2.resize ที่มี interpolation ให้",
        "good_for": ["cameraman.png", "A3.png"],
        "notes": [("info", "ขยาย (>1) จะเกิดรูโหว่เพราะ forward map ไม่มี interpolation")],
        "code": "scale",
    },
    {
        "id": "rotate", "name": "Rotation (หมุนรอบมุมซ้ายบน)", "chapter": 4, "stars": 4,
        "fn": transform.rotate, "input": "any",
        "params": [{"key": "theta", "type": "float", "min": -180, "max": 180, "step": 5,
                    "default": -45, "label": "θ (องศา)"}],
        "formula": "R(θ) = [[cosθ, sinθ, 0], [−sinθ, cosθ, 0], [0, 0, 1]]",
        "why": "หมุนรอบจุด (0,0) — ภาพส่วนใหญ่จะหลุดกรอบ ต้องใช้ Composite ถึงจะหมุนรอบกลางภาพ",
        "good_for": ["cameraman.png", "A3.png"],
        "notes": [
            ("warn", "⚠️ สไลด์ 2 ไฟล์เขียนเครื่องหมาย sin ตรงข้ามกัน! เวอร์ชันภาพเทา "
                     "(image_spatial_transformation) ใช้ R[0,1]=+sin, R[1,0]=−sin ส่วนเวอร์ชันสี "
                     "(..._color) ใช้ R[0,1]=−sin, R[1,0]=+sin เว็บนี้ยึดเวอร์ชันภาพเทา "
                     "ถ้าข้อสอบให้คำนวณมือ ต้องดูให้ดีว่าโจทย์อ้างอิงอันไหน"),
            ("info", "อย่าลืมแปลงองศาเป็นเรเดียนก่อน: rad = θ × π / 180"),
        ],
        "code": "rotate",
    },
    {
        "id": "composite", "name": "Composite (หมุน+ย่อ รอบกึ่งกลาง)", "chapter": 4, "stars": 4,
        "fn": transform.composite, "input": "any",
        "params": [
            {"key": "theta", "type": "float", "min": -180, "max": 180, "step": 5, "default": -45,
             "label": "θ (องศา)"},
            {"key": "sx", "type": "float", "min": 0.2, "max": 2.0, "step": 0.05, "default": 0.8,
             "label": "sx"},
            {"key": "sy", "type": "float", "min": 0.2, "max": 2.0, "step": 0.05, "default": 0.8,
             "label": "sy"},
        ],
        "formula": "T = Ts(−xc,−yc) · R(θ) · S(sx,sy) · Ts(xc,yc)",
        "why": "สำนวนที่อาจารย์เน้นที่สุดในบทนี้ — หมุนภาพรอบจุดกึ่งกลางอย่างถูกวิธี",
        "good_for": ["cameraman.png", "A3.png", "lena_color_256.png"],
        "notes": [
            ("info", "ลำดับสำคัญมาก: เลื่อนกึ่งกลางไป (0,0) → หมุน/ย่อ → เลื่อนกลับ"),
            ("warn", "ผลลัพธ์จะมีรูโหว่เป็นจุดดำ เพราะ forward map + int() ตัดเศษ — "
                     "นี่คือของจริงตามสไลด์ ไม่ใช่บั๊ก สไลด์แก้ด้วย cv2.medianBlur(out, 3) ทีหลัง "
                     "ลองต่อ Median Filter ในโหมด pipeline ดูได้"),
        ],
        "code": "composite",
    },

    # ------------------------------------------------------------------ ch.5
    {
        "id": "mean", "name": "Mean / Average Filter", "chapter": 5, "stars": 5,
        "fn": filtering.mean_filter, "input": "gray",
        "params": [{"key": "k", "type": "odd_int", "min": 3, "max": 9, "step": 2, "default": 3,
                    "label": "k (ขนาดเคอร์เนล)"}],
        "formula": "s = (1/k²) · Σ r_i",
        "why": "ลบนอยส์แบบ Gaussian ได้บ้าง แต่ทำให้ภาพเบลอ — ใช้เทียบกับ Median ให้เห็นข้อเสีย",
        "good_for": ["lena_noise_512.png", "lena_salt_512.png", "lena_speckle_512.png"],
        "notes": [
            ("info", "k ต้องเป็นเลขคี่เสมอ"),
            ("warn", "สไลด์เขียน mask = np.ones([k,k])/9 — หาร 9 ตายตัว ซึ่งผิดเมื่อ k≠3 "
                     "ที่ถูกต้องหาร k² เว็บนี้ใช้ /(k*k)"),
            ("info", "ลองกับ lena_salt แล้วเทียบกับ Median — จุด salt จะไม่หาย แค่ถูกเกลี่ยเปื้อน"),
            LOOP_NOTE,
        ],
        "code": "mean",
    },
    {
        "id": "median", "name": "Median Filter", "chapter": 5, "stars": 5,
        "fn": filtering.median_filter, "input": "gray",
        "params": [{"key": "k", "type": "odd_int", "min": 3, "max": 9, "step": 2, "default": 3,
                    "label": "k (ขนาดเคอร์เนล)"}],
        "formula": "s = median{ r_i }",
        "why": "คำตอบมาตรฐานของ Salt & Pepper Noise — ตัดค่าสุดโต่งทิ้งได้สนิทโดยภาพไม่เบลอ",
        "good_for": ["lena_salt_512.png", "lena_noise_512.png", "lena_speckle_512.png"],
        "notes": [
            ("info", "ต้องวนลูปแล้ว sort ก่อนหยิบตัวกลาง ตำแหน่ง median = ⌊k²/2⌋ (k=3 → index 4)"),
            ("warn", "สไลด์เขียน MedianBlur(img, k) แต่ข้างในเรียก cv2.medianBlur(img, 3) — "
                     "hardcode 3 ทำให้ k ที่ส่งไปไม่มีผล เว็บนี้ส่ง k จริง"),
            LOOP_NOTE,
        ],
        "code": "median",
    },
    {
        "id": "edge", "name": "Prewitt / Sobel (edge_operator)", "chapter": 5, "stars": 5,
        "fn": filtering.edge_operator, "input": "gray",
        "params": [{"key": "k", "type": "choice", "default": 2, "label": "k",
                    "options": [[1, "k=1 → Prewitt"], [2, "k=2 → Sobel"]]}],
        "formula": "Gx = [[−1,0,1],[−k,0,k],[−1,0,1]],  G = √(Gx² + Gy²)",
        "why": "หาขอบวัตถุ — Sobel (k=2) ให้น้ำหนักแถวกลาง 2 เท่า จึงทนนอยส์กว่า Prewitt",
        "good_for": ["cameraman.png", "A3.png", "lena.png"],
        "notes": [
            ("info", "อาจารย์รวม Prewitt กับ Sobel เป็นฟังก์ชันเดียวโดยใช้ k เป็นตัวแปร — "
                     "จำสูตรเดียวตอบได้ทั้ง 2 ข้อ"),
            ("info", "ถ้าภาพมีนอยส์ ให้ Median ก่อนแล้วค่อยหาขอบ (ลองในโหมด pipeline)"),
            LOOP_NOTE,
        ],
        "code": "edge",
    },
    {
        "id": "canny", "name": "Canny Edge Detection", "chapter": 5, "stars": 2,
        "fn": filtering.canny, "input": "gray",
        "params": [
            {"key": "low", "type": "int", "min": 0, "max": 255, "step": 5, "default": 50,
             "label": "low threshold"},
            {"key": "high", "type": "int", "min": 0, "max": 255, "step": 5, "default": 150,
             "label": "high threshold"},
        ],
        "formula": "cv2.Canny(img, low, high)",
        "why": "ขอบบางคมกว่า Sobel — แต่อาจารย์สอนแค่การเรียกใช้ ไม่ได้ลงอัลกอริทึม",
        "good_for": ["cameraman.png", "A3.png", "lena.png"],
        "notes": [("info", "ค่ามาตรฐานในสไลด์คือ 50 / 150")],
        "code": "canny",
    },

    # ------------------------------------------------------------------ ch.6
    {
        "id": "binarize", "name": "Global Threshold (เลือก T เอง)", "chapter": 6, "stars": 4,
        "fn": threshold.binarize, "input": "gray",
        "params": [{"key": "t", "type": "int", "min": 0, "max": 255, "step": 1, "default": 100,
                    "label": "T"}],
        "formula": "out = 255 ถ้า r ≥ T, ไม่งั้น 0",
        "why": "พื้นฐานที่สุด — ลากสไลเดอร์ดูว่าทำไมการเลือก T ด้วยมือถึงยาก แล้วค่อยไปดู Otsu",
        "good_for": ["bank4.jpg", "cells.tif", "document1.png"],
        "notes": [
            ("info", "เงื่อนไขคือ ≥ ไม่ใช่ > (พิกเซลที่เท่ากับ T พอดี จะเป็นสีขาว)"),
            ("info", "ลากสไลเดอร์แล้วดู histogram ด้านล่าง — T ที่ดีควรตกในหุบเขาระหว่าง 2 ยอด"),
            LOOP_NOTE,
        ],
        "code": "binarize",
    },
    {
        "id": "otsu", "name": "Otsu's Method", "chapter": 6, "stars": 5,
        "fn": threshold.otsu, "input": "gray", "params": [],
        "formula": "T = argmax_t  w0(t)·w1(t)·(μ0(t) − μ1(t))²",
        "why": "หา T อัตโนมัติ — เหมาะกับภาพที่ histogram มี 2 ยอดชัด (วัตถุขาว/พื้นหลังดำ)",
        "good_for": ["cells.tif", "cell.tif", "bank4.jpg", "cameraman.png"],
        "notes": [
            ("info", "ทำให้ความแปรปรวนระหว่างคลาสสูงสุด = ทำให้ภายในคลาสต่ำสุด "
                     "เพราะ σ²_T = σ²_W + σ²_B คงที่"),
            ("warn", "ลูป t ต้องเริ่มที่ 1 และจบที่ 254 (range(1,256)) ห้ามเริ่ม 0 หรือจบ 255 "
                     "ไม่งั้น w0 หรือ w1 เป็นศูนย์ → หารศูนย์ → NaN"),
            ("info", "ตัวกันหารศูนย์ +0.00000001 คือลายเซ็นของโค้ดอาจารย์ ใส่ไว้ให้เหมือน"),
            ("info", "ถ้ามีหลาย t ได้คะแนนเท่ากัน เงื่อนไข coef > coef_max (มากกว่าเฉยๆ) "
                     "จะเลือกตัวแรกเสมอ — ตรงกับตัวอย่างในสรุปที่ t=3,4,5 เท่ากัน แล้วตอบ 3"),
            GTE_NOTE,
        ],
        "code": "otsu",
    },
    {
        "id": "intermean", "name": "Intermean (วนซ้ำ)", "chapter": 6, "stars": 5,
        "fn": threshold.intermean, "input": "gray",
        "params": [{"key": "t0", "type": "int", "min": 0, "max": 255, "step": 1, "default": 128,
                    "label": "T₀ (ค่าเริ่มต้น)"}],
        "formula": "T_{n+1} = round( (μ0 + μ1) / 2 ),  หยุดเมื่อ |T_{n+1} − T_n| ≤ 1",
        "why": "หา T อัตโนมัติแบบวนซ้ำจนลู่เข้า — คิดง่ายกว่า Otsu และคำนวณมือได้ในห้องสอบ",
        "good_for": ["gray2.png", "cells.tif", "bank4.jpg"],
        "notes": [
            ("info", "ลองเปลี่ยน T₀ ดู — จะเห็นว่าลู่เข้าค่าเดิมเกือบเสมอ ไม่ว่าเริ่มจากไหน "
                     "นี่คือจุดแข็งของวิธีนี้"),
            ("info", "ค่าตั้งต้นมาตรฐานคือ int(np.mean(img)) หรือ 128"),
            ("warn", "ต้องใช้การปัดเศษแบบ round-half-up (float2int ที่อาจารย์เขียนเอง) "
                     "ไม่ใช่ round() ของ Python ที่ปัด 2.5 เป็น 2 (banker's rounding)"),
            GTE_NOTE,
        ],
        "code": "intermean",
    },
    {
        "id": "intermean_loop", "name": "Intermean Loop (หลายระดับ)", "chapter": 6, "stars": 3,
        "fn": threshold.intermean_loop, "input": "gray",
        "params": [{"key": "times", "type": "int", "min": 1, "max": 4, "step": 1, "default": 3,
                    "label": "จำนวนรอบ"}],
        "formula": "หา T ในหน้าต่าง [st, en) → ตั้ง st = T+1 → ทำซ้ำ",
        "why": "ภาพที่มีมากกว่า 2 กลุ่มความสว่าง — แบ่งซ้อนไปเรื่อยๆ ทีละระดับ",
        "good_for": ["gray2.png", "gray3.png", "cells.tif"],
        "notes": [("info", "แต่ละรอบตัดช่วงล่างทิ้งแล้วหา T ใหม่เฉพาะช่วงบน")],
        "code": "intermean_loop",
    },
    {
        "id": "split_view", "name": "split4 (ดูการแบ่ง 4 ส่วน)", "chapter": 6, "stars": 4,
        "fn": threshold.split_view, "input": "gray", "params": [],
        "formula": "h_half = h // 2,  w_half = w // 2",
        "why": "เห็นภาพว่า split4 ตัดตรงไหน ก่อนไปใช้กับ Local Adaptive",
        "good_for": ["gray3.png", "document2.jpg", "document1.png", "shade.png"],
        "notes": [
            ("info", "ไม่ได้อยู่ใน slide Aj — มาจาก Boss/week7 real/Local_splits_method.ipynb"),
            ("info", "ถ้า h หรือ w เป็นเลขคี่ ส่วนล่าง/ขวาจะใหญ่กว่า 1 พิกเซล เพราะ slice ท้ายเก็บเศษ"),
        ],
        "code": "split_view",
    },
    {
        "id": "local_adaptive", "name": "Local Splits Adaptive Threshold", "chapter": 6, "stars": 4,
        "fn": threshold.local_adaptive, "input": "gray",
        "params": [{"key": "method", "type": "choice", "default": "otsu", "label": "วิธีหา T แต่ละส่วน",
                    "options": [["otsu", "Otsu"], ["intermean", "Intermean"]]}],
        "formula": "split4 → หา T_i แยกกันแต่ละส่วน → MergeLocal",
        "why": "ภาพถ่ายที่มีเงาพาดผ่านทำให้มืดไม่เท่ากัน — T ค่าเดียวใช้ทั้งภาพไม่ได้",
        "good_for": ["gray3.png", "document2.jpg", "document1.png", "shade.png"],
        "notes": [
            ("info", "ไม่ได้อยู่ใน slide Aj — ประกอบจาก Boss/week7 real/ "
                     "(Local_splits_method + Otsu_adaptation)"),
            ("info", "ลองกับ gray3.png (รูปที่อาจารย์ใช้เอง) แล้วกด Otsu ธรรมดาเทียบดู — "
                     "Otsu แบบ global จะพังสิ้นเชิง คำว่า 'python' กับ 'images' หายไปกลายเป็น "
                     "บล็อกขาว/ดำล้วน เพราะแต่ละส่วนพื้นหลังสว่างไม่เท่ากัน แต่ local split4 "
                     "อ่านออกครบทั้ง 4 คำ นี่คือเหตุผลทั้งหมดของวิธีนี้"),
            ("info", "ดูตาราง T ของแต่ละส่วนด้านล่าง — ยิ่งค่าห่างกันมาก ยิ่งแปลว่าภาพแสงไม่สม่ำเสมอ "
                     "และยิ่งคุ้มที่จะใช้ local แทน global (gray3.png ห่างกันถึง 125)"),
            GTE_NOTE,
        ],
        "code": "local_adaptive",
    },
]

BY_ID = {f["id"]: f for f in FUNCTIONS}


def public_catalog():
    """The registry minus the un-serializable callables."""
    out = []
    for f in FUNCTIONS:
        d = {k: v for k, v in f.items() if k != "fn"}
        d["notes"] = [{"level": lv, "text": tx} for lv, tx in f.get("notes", [])]
        out.append(d)
    return out
