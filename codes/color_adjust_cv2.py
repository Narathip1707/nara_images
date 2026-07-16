import cv2
import numpy as np


def rgb2hsv(img):
    """เวอร์ชัน vectorized -- เร็วกว่า for-loop มาก แต่ผลลัพธ์เท่ากันเป๊ะ"""
    b, g, r = cv2.split(img / 255.)
    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    delta = c_max - c_min
    h = np.zeros_like(r, dtype="float32")

    mask_delta = delta != 0
    mask_r = mask_delta & (c_max == r)
    mask_g = mask_delta & (c_max == g)
    mask_b = mask_delta & (c_max == b)

    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    s = np.zeros_like(h, dtype="float32")
    mask_v = c_max != 0
    s[mask_v] = delta[mask_v] / c_max[mask_v]
    v = c_max
    return np.stack([h, s, v], axis=2)


def saturation_adjust(s, factor, mask):
    s[mask] = s[mask] * factor
    s = np.clip(s, 0, 1.0)
    return s


def value_adjust(v, factor, mask):
    v[mask] = v[mask] * factor
    v = np.clip(v, 0, 1.0)
    return v


def hsv2bgr(hsv_img):
    return (cv2.cvtColor(hsv_img.astype("float32"), cv2.COLOR_HSV2BGR) * 255).astype("uint8")


def color_adjust(img, bands, s_factor=1.0, v_factor=1.0):
    h, s, v = cv2.split(rgb2hsv(img))
    mask = np.zeros_like(h, dtype=bool)
    for low, high in bands:
        mask |= (h >= low) | (h <= high) if low > high else (h >= low) & (h <= high)
    s = saturation_adjust(s, s_factor, mask)
    v = value_adjust(v, v_factor, mask)
    return hsv2bgr(np.stack([h, s, v], axis=2))


# หมายเหตุ: นี่คือโค้ดที่เราเขียนตอบไฟนอลข้อ 2 เอง (Final_test/6601001123_Q2/image_tool.py)
# ของจริงในไฟนอลใช้แบบนี้:
#   masks = [(h > 180) & (h < 260), (h > 20) & (h < 60)]   # ฟ้ากับส้ม
#   s = saturation_adjust(s, 0.2, masks[0])                 # ฟ้าซีดลง
#   s = saturation_adjust(s, 40.0, masks[1])                # ส้มสดขึ้น (40 ก็แค่ชน clip ที่ 1.0)
#   v = value_adjust(v, 1.3, masks[1])                      # ส้มสว่างขึ้น
# ต่างจาก color_adjust_loop.py 2 จุด:
#   1) hsv2bgr ผ่าน cv2 เพี้ยนจากสูตรมือ +-1 บนพิกเซลราว 27% (for-loop ตามไม่ได้)
#   2) ไฟนอลใช้ > < แต่สไลด์ cell 12 ใช้ >= <= -- แอปนี้ยึดสไลด์
if __name__ == "__main__":
    img = cv2.imread("./images/lena_color_256.png")
    out = color_adjust(img, [(200, 260)], s_factor=0.2)
    cv2.imshow("color adjust", cv2.hconcat([img, out]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
