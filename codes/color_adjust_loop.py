import cv2
import numpy as np


def rgb2hsv(img):
    rows, cols = img.shape[0], img.shape[1]
    h = np.zeros((rows, cols), np.float64)
    s = np.zeros((rows, cols), np.float64)
    v = np.zeros((rows, cols), np.float64)
    for y in range(rows):
        for x in range(cols):
            b = img[y, x, 0] / 255.0
            g = img[y, x, 1] / 255.0
            r = img[y, x, 2] / 255.0
            cmax = max(r, g, b)
            cmin = min(r, g, b)
            delta = cmax - cmin

            if delta == 0:
                h[y, x] = 0
            elif cmax == r:
                h[y, x] = 60 * (((g - b) / delta) % 6)
            elif cmax == g:
                h[y, x] = 60 * (((b - r) / delta) + 2)
            else:
                h[y, x] = 60 * (((r - g) / delta) + 4)

            s[y, x] = 0 if cmax == 0 else delta / cmax
            v[y, x] = cmax
    return h, s, v


def hsv2bgr(h, s, v):
    rows, cols = h.shape
    out = np.zeros((rows, cols, 3), np.uint8)
    for y in range(rows):
        for x in range(cols):
            hh, ss, vv = h[y, x], s[y, x], v[y, x]
            c = vv * ss
            xx = c * (1 - abs(((hh / 60.0) % 2) - 1))
            m = vv - c

            i = int(hh // 60) % 6
            if i == 0:
                r, g, b = c, xx, 0
            elif i == 1:
                r, g, b = xx, c, 0
            elif i == 2:
                r, g, b = 0, c, xx
            elif i == 3:
                r, g, b = 0, xx, c
            elif i == 4:
                r, g, b = xx, 0, c
            else:
                r, g, b = c, 0, xx

            out[y, x, 0] = int(round((b + m) * 255))
            out[y, x, 1] = int(round((g + m) * 255))
            out[y, x, 2] = int(round((r + m) * 255))
    return out


def in_band(hue, low, high):
    if high - low >= 360 or high - low <= -360:
        return True  # เลือกทั้งวงกลมสี
    low = low % 360
    high = high % 360
    if low <= high:
        return low <= hue <= high  # ช่วงปกติ เช่น 20-60
    # low > high แปลว่าช่วงวนผ่าน 0 องศา เช่น 340-20 (สีแดง) ต้องใช้ or ไม่ใช่ and
    return hue >= low or hue <= high


def color_adjust(img, bands, s_min=0.25, v_min=0.15,
                 s_factor=1.0, v_factor=1.0, h_shift=0.0):
    rows, cols = img.shape[0], img.shape[1]
    h, s, v = rgb2hsv(img)

    for y in range(rows):
        for x in range(cols):
            hit = False
            for low, high in bands:
                if in_band(h[y, x], low, high) and s[y, x] >= s_min and v[y, x] >= v_min:
                    hit = True
                    break
            if not hit:
                continue  # อยู่นอกช่วงสีที่เลือก ปล่อยไว้เหมือนเดิม

            # S กับ V ต้อง clip [0,1] ทุกครั้ง ไม่งั้นค่าล้นตอนแปลงกลับ
            s[y, x] = min(max(s[y, x] * s_factor, 0.0), 1.0)
            v[y, x] = min(max(v[y, x] * v_factor, 0.0), 1.0)
            h[y, x] = (h[y, x] + h_shift) % 360  # H เป็นวงกลม ต้อง mod 360

    return hsv2bgr(h, s, v)


if __name__ == "__main__":
    img = cv2.imread("./images/lena_color_256.png")
    # ลดความสดสีฟ้าลงเหลือ 20% แบบไฟนอลข้อ 2
    out = color_adjust(img, [(200, 260)], s_factor=0.2)
    cv2.imshow("color adjust", cv2.hconcat([img, out]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
