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


def in_band(hue, low, high):
    if high - low >= 360 or high - low <= -360:
        return True  # เลือกทั้งวงกลมสี
    low = low % 360
    high = high % 360
    if low <= high:
        return low <= hue <= high  # ช่วงปกติ เช่น 20-60
    # low > high แปลว่าช่วงวนผ่าน 0 องศา เช่น 340-20 (สีแดง) ต้องใช้ or ไม่ใช่ and
    return hue >= low or hue <= high


def color_select(img, bands, s_min=0.25, v_min=0.15, drop="gray"):
    rows, cols = img.shape[0], img.shape[1]
    h, s, v = rgb2hsv(img)

    out = np.zeros((rows, cols, 3), np.uint8)
    for y in range(rows):
        for x in range(cols):
            keep = False
            for low, high in bands:
                # ต้องผ่านทั้ง 3 เงื่อนไข: อยู่ในช่วงสี + สีสดพอ + สว่างพอ
                if in_band(h[y, x], low, high) and s[y, x] >= s_min and v[y, x] >= v_min:
                    keep = True
                    break

            if keep:
                out[y, x] = img[y, x]
            elif drop == "gray":
                # ที่เหลือเป็นขาวดำ = color pop
                b, g, r = img[y, x, 0], img[y, x, 1], img[y, x, 2]
                gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                out[y, x] = int(gray)
            elif drop == "white":
                out[y, x] = 255
            else:
                out[y, x] = 0  # drop == "black" แบบสไลด์อาจารย์
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/lena_color_256.png")
    # เลือกฟ้า 200-260 กับ ส้ม 20-60 พร้อมกัน แบบไฟนอลข้อ 2
    out = color_select(img, [(200, 260), (20, 60)])
    cv2.imshow("color select", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
