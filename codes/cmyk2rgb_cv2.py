import cv2
import numpy as np


def cmyk2rgb(c, m, y, k):
    r = 255 * (1 - c) * (1 - k)
    g = 255 * (1 - m) * (1 - k)
    b = 255 * (1 - y) * (1 - k)
    return cv2.merge([b, g, r]).astype(np.uint8)


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
