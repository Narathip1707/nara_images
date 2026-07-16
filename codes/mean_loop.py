import cv2
import numpy as np


def mean_filter(img, k):
    rows, cols = img.shape
    bd = k // 2                 # ขอบที่ต้องเว้น
    out = img.copy()            # copy ไว้ ขอบภาพจึงไม่กลายเป็นดำ
    for y in range(bd, rows - bd):
        for x in range(bd, cols - bd):
            total = 0
            for i in range(-bd, bd + 1):
                for j in range(-bd, bd + 1):
                    total += int(img[y + i, x + j])
            out[y, x] = total / (k * k)
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/lena_noise_512.png", 0)
    out = mean_filter(img, 3)
    cv2.imshow("mean", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
