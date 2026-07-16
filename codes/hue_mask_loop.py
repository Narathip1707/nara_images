import cv2
import numpy as np


def get_hue(img):
    rows, cols = img.shape[0], img.shape[1]
    h = np.zeros((rows, cols), np.float64)
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
    return h


def hue_mask(img, low, high):
    h = get_hue(img)
    rows, cols = img.shape[0], img.shape[1]
    out = np.zeros((rows, cols, 3), np.uint8)
    for y in range(rows):
        for x in range(cols):
            # เก็บเฉพาะ pixel ที่ Hue อยู่ในช่วง นอกนั้นปล่อยเป็นดำ
            if low <= h[y, x] <= high:
                out[y, x] = img[y, x]
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/RGB_test.png")
    out = hue_mask(img, 0, 60)  # ช่วงแดง-เหลือง (องศา)
    cv2.imshow("hue mask", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
