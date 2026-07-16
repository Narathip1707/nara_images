import cv2
import numpy as np


def rgb2cmyk(img):
    b = img[:, :, 0] / 255.0
    g = img[:, :, 1] / 255.0
    r = img[:, :, 2] / 255.0
    c1 = 1 - r
    m1 = 1 - g
    y1 = 1 - b
    k = np.minimum(np.minimum(c1, m1), y1)
    den = 1 - k
    den[den == 0] = 1  # กัน หาร 0 ตอน K == 1
    c = (c1 - k) / den
    m = (m1 - k) / den
    y = (y1 - k) / den
    return c, m, y, k


if __name__ == "__main__":
    img = cv2.imread("./images/lena_color_256.png")
    c, m, y, k = rgb2cmyk(img)
    cv2.imshow("K", (k * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
