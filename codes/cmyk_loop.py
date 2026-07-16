import cv2
import numpy as np


def rgb2cmyk(img):
    rows, cols = img.shape[0], img.shape[1]
    c = np.zeros((rows, cols), np.float64)
    m = np.zeros((rows, cols), np.float64)
    yy = np.zeros((rows, cols), np.float64)
    k = np.zeros((rows, cols), np.float64)
    for y in range(rows):
        for x in range(cols):
            # normalize เป็น [0,1] ก่อนเสมอ
            b = img[y, x, 0] / 255.0
            g = img[y, x, 1] / 255.0
            r = img[y, x, 2] / 255.0
            c1 = 1 - r
            m1 = 1 - g
            y1 = 1 - b
            kk = min(c1, m1, y1)
            if kk == 1:
                # จุดสีดำสนิท ตัวหารเป็น 0 ต้องดักไว้
                c[y, x] = 0
                m[y, x] = 0
                yy[y, x] = 0
            else:
                c[y, x] = (c1 - kk) / (1 - kk)
                m[y, x] = (m1 - kk) / (1 - kk)
                yy[y, x] = (y1 - kk) / (1 - kk)
            k[y, x] = kk
    return c, m, yy, k


if __name__ == "__main__":
    img = cv2.imread("./images/lena_color_256.png")
    c, m, y, k = rgb2cmyk(img)
    cv2.imshow("C", (c * 255).astype(np.uint8))
    cv2.imshow("M", (m * 255).astype(np.uint8))
    cv2.imshow("Y", (y * 255).astype(np.uint8))
    cv2.imshow("K", (k * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
