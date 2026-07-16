import cv2
import numpy as np


def transpose(img):
    rows, cols = img.shape
    # ภาพผลลัพธ์สลับขนาดเป็น (cols, rows)
    out = np.zeros((cols, rows), np.uint8)
    for y in range(cols):
        for x in range(rows):
            out[y, x] = img[x, y]
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    out = transpose(img)
    cv2.imshow("transpose", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
