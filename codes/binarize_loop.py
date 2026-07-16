import cv2
import numpy as np


def binarize(img, T):
    rows, cols = img.shape
    out = np.zeros((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            out[y, x] = 255 if img[y, x] >= T else 0
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/document1.png", 0)
    out = binarize(img, 128)
    cv2.imshow("binarize", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
