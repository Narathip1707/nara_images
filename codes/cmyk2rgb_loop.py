import cv2
import numpy as np


def cmyk2rgb(c, m, y, k):
    rows, cols = c.shape[0], c.shape[1]
    out = np.zeros((rows, cols, 3), np.uint8)
    for yy in range(rows):
        for x in range(cols):
            r = 255 * (1 - c[yy, x]) * (1 - k[yy, x])
            g = 255 * (1 - m[yy, x]) * (1 - k[yy, x])
            b = 255 * (1 - y[yy, x]) * (1 - k[yy, x])
            # เก็บกลับเป็น BGR ให้ cv2 แสดงผลถูกสี
            out[yy, x, 0] = b
            out[yy, x, 1] = g
            out[yy, x, 2] = r
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/lena_color_256.png")
    b = img[:, :, 0] / 255.0
    g = img[:, :, 1] / 255.0
    r = img[:, :, 2] / 255.0
    c1, m1, y1 = 1 - r, 1 - g, 1 - b
    k = np.minimum(np.minimum(c1, m1), y1)
    den = 1 - k
    den[den == 0] = 1
    c, m, y = (c1 - k) / den, (m1 - k) / den, (y1 - k) / den

    back = cmyk2rgb(c, m, y, k)
    cv2.imshow("back to rgb", back)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
