import cv2
import numpy as np


def flip(img, opt):
    rows, cols = img.shape
    out = np.zeros((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            if opt == 0:                       # ซ้าย-ขวา
                out[y, x] = img[y, cols - 1 - x]
            elif opt == 1:                     # บน-ล่าง
                out[y, x] = img[rows - 1 - y, x]
            else:                              # ทั้งสองแกน
                out[y, x] = img[rows - 1 - y, cols - 1 - x]
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    cv2.imshow("horizontal", flip(img, 0))
    cv2.imshow("vertical", flip(img, 1))
    cv2.imshow("both", flip(img, 2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
