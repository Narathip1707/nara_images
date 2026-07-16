import cv2
import numpy as np


def negative(img):
    rows, cols = img.shape
    out = np.zeros((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            out[y, x] = 255 - img[y, x]
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/breast.bmp", 0)
    out = negative(img)
    cv2.imshow("negative", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
