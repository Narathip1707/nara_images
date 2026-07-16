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
    # กับดัก: S กับ V เป็นทศนิยม [0,1] ห้าม int() ไม่งั้นกลายเป็น 0 หมด
    return h, s, v


if __name__ == "__main__":
    img = cv2.imread("./images/lena_color_256.png")
    h, s, v = rgb2hsv(img)
    cv2.imshow("H", (h / 360 * 255).astype(np.uint8))
    cv2.imshow("S", (s * 255).astype(np.uint8))
    cv2.imshow("V", (v * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
