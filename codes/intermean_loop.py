import cv2
import numpy as np
import math


def float2int(x):
    # ปัดขึ้นเมื่อเศษ >= 0.5 (Python round() ปัดเข้าเลขคู่ ใช้ไม่ได้)
    return math.ceil(x) if x - math.floor(x) >= 0.5 else math.floor(x)


def histogram(img):
    rows, cols = img.shape
    hist = np.zeros(256, np.int64)
    for y in range(rows):
        for x in range(cols):
            hist[img[y, x]] += 1
    return hist


def intermean(img):
    hist = histogram(img)
    prob = hist / np.sum(hist)
    t0 = -999
    t1 = float2int(np.sum(np.arange(0, 256) * prob))  # เริ่มที่ค่าเฉลี่ยทั้งภาพ
    while abs(t1 - t0) > 1:                           # ลู่เข้าเมื่อต่างกันไม่เกิน 1
        t0 = t1
        w0 = np.sum(prob[:t0]) + 0.00000001
        w1 = np.sum(prob[t0:]) + 0.00000001
        u0 = np.sum(np.arange(0, t0) * prob[:t0]) / w0
        u1 = np.sum(np.arange(t0, 256) * prob[t0:]) / w1
        t1 = float2int((u0 + u1) / 2)
    return t1


def binarize(img, T):
    rows, cols = img.shape
    out = np.zeros((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            out[y, x] = 255 if img[y, x] >= T else 0
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/document1.png", 0)
    T = intermean(img)
    print("threshold =", T)
    cv2.imshow("intermean", binarize(img, T))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
