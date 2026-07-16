import cv2
import numpy as np


def median_filter(img, k):
    rows, cols = img.shape
    bd = k // 2
    inx = (k * k) // 2          # ตำแหน่งค่ากลางหลังเรียง
    out = img.copy()
    for y in range(bd, rows - bd):
        for x in range(bd, cols - bd):
            win = []
            for i in range(-bd, bd + 1):
                for j in range(-bd, bd + 1):
                    win.append(img[y + i, x + j])
            win.sort()
            out[y, x] = win[inx]
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/lena_salt_512.png", 0)
    out = median_filter(img, 3)  # median เก่ง salt & pepper กว่า mean
    cv2.imshow("median", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
