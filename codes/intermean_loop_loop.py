import cv2
import numpy as np
import math


def float2int(x):
    return math.ceil(x) if x - math.floor(x) >= 0.5 else math.floor(x)


def histogram(img):
    rows, cols = img.shape
    hist = np.zeros(256, np.int64)
    for y in range(rows):
        for x in range(cols):
            hist[img[y, x]] += 1
    return hist


def intermean_window(prob, st, en):
    # หา threshold เฉพาะในช่วง [st, en) ไม่ใช่ทั้ง 0-255
    t0 = -999
    t1 = float2int((st + en) / 2)
    while abs(t1 - t0) > 1:
        t0 = t1
        w0 = np.sum(prob[st:t0]) + 0.00000001
        w1 = np.sum(prob[t0:en]) + 0.00000001
        u0 = np.sum(np.arange(st, t0) * prob[st:t0]) / w0
        u1 = np.sum(np.arange(t0, en) * prob[t0:en]) / w1
        t1 = float2int((u0 + u1) / 2)
    return t1


def intermean_loop(img, times):
    hist = histogram(img)
    prob = hist / np.sum(hist)
    st = 0
    en = 256
    thrs = []
    for i in range(times):
        thr = intermean_window(prob, st, en)
        thrs.append(thr)
        st = thr + 1   # รอบถัดไปหาต่อทางขวาของ threshold เดิม
    return thrs


if __name__ == "__main__":
    img = cv2.imread("./images/cell.tif", 0)
    thrs = intermean_loop(img, 3)
    print("thresholds =", thrs)
