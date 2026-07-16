import cv2
import numpy as np


def histogram(img):
    rows, cols = img.shape
    hist = np.zeros(256, np.int64)
    for y in range(rows):
        for x in range(cols):
            hist[img[y, x]] += 1
    return hist


def otsu(img):
    hist = histogram(img)
    prob = hist / np.sum(hist)
    mx = 0
    thr = 0
    for t in range(1, 256):
        w0 = np.sum(prob[:t]) + 0.00000001   # กันหารศูนย์
        w1 = np.sum(prob[t:]) + 0.00000001
        u0 = np.sum(np.arange(0, t) * prob[:t]) / w0
        u1 = np.sum(np.arange(t, 256) * prob[t:]) / w1
        coef = w0 * w1 * (u0 - u1) ** 2      # between-class variance
        if coef > mx:                        # ใช้ > เท่านั้น ได้ค่า t ตัวแรกที่สูงสุด
            mx = coef
            thr = t
    return thr


def binarize(img, T):
    rows, cols = img.shape
    out = np.zeros((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            out[y, x] = 255 if img[y, x] >= T else 0
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/document1.png", 0)
    T = otsu(img)
    print("threshold =", T)
    cv2.imshow("otsu", binarize(img, T))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
