import cv2
import numpy as np


def to_gray(img):
    rows, cols = img.shape[0], img.shape[1]
    out = np.zeros((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            # cv2.imread อ่านมาเป็น BGR ไม่ใช่ RGB : 0=B, 1=G, 2=R
            b = img[y, x, 0]
            g = img[y, x, 1]
            r = img[y, x, 2]
            out[y, x] = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/lena_color_256.png")
    gray = to_gray(img)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
