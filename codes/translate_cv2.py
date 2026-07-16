import cv2
import numpy as np


def translate(img, tx, ty):
    rows, cols = img.shape
    # กับดัก: warpAffine ใช้เมทริกซ์ 2x3 แบบ column-vector (สลับกับในสไลด์)
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    return cv2.warpAffine(img, M, (cols, rows))


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    out = translate(img, 50, 30)
    cv2.imshow("translate", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
