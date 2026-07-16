import cv2
import numpy as np


def histogram(img):
    rows, cols = img.shape
    hist = np.zeros(256, np.int64)
    for y in range(rows):
        for x in range(cols):
            hist[img[y, x]] += 1
    return hist


def equalize(img):
    hist = histogram(img)
    cdf = hist.cumsum()
    # mask ค่า 0 ทิ้ง ไม่ให้ระดับสีที่ไม่มีในภาพมาดึง min ต่ำเกินจริง
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = ((cdf_m - cdf_m.min()) / (cdf_m.max() - cdf_m.min())) * 255
    cdf_m = np.ma.filled(cdf_m, 0).astype('uint8')

    rows, cols = img.shape
    out = np.zeros((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            out[y, x] = cdf_m[img[y, x]]  # ใช้ cdf เป็นตาราง lookup
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/dark2.png", 0)
    out = equalize(img)
    cv2.imshow("equalize", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
