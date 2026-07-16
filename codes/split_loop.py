import cv2
import numpy as np


def split_channel(img):
    rows, cols = img.shape[0], img.shape[1]
    b = np.zeros((rows, cols), np.uint8)
    g = np.zeros((rows, cols), np.uint8)
    r = np.zeros((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            b[y, x] = img[y, x, 0]
            g[y, x] = img[y, x, 1]
            r[y, x] = img[y, x, 2]
    return b, g, r


def stack_vertical(b, g, r):
    # ต่อภาพในแนวตั้ง (แทน cv2.vconcat) -> สูงเป็น 3 เท่า
    return np.concatenate((b, g, r), axis=0)


if __name__ == "__main__":
    img = cv2.imread("./images/lena_color_256.png")
    b, g, r = split_channel(img)
    out = stack_vertical(b, g, r)
    cv2.imshow("b g r", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
