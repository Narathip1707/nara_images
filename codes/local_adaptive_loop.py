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
        w0 = np.sum(prob[:t]) + 0.00000001
        w1 = np.sum(prob[t:]) + 0.00000001
        u0 = np.sum(np.arange(0, t) * prob[:t]) / w0
        u1 = np.sum(np.arange(t, 256) * prob[t:]) / w1
        coef = w0 * w1 * (u0 - u1) ** 2
        if coef > mx:
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


def split4(img):
    h, w = img.shape
    h_half = h // 2
    w_half = w // 2
    return [img[:h_half, :w_half], img[:h_half, w_half:],
            img[h_half:, :w_half], img[h_half:, w_half:]]


def MergeLocal(x, out):
    h, w = out.shape
    h_half = h // 2
    w_half = w // 2
    out[:h_half, :w_half] = x[0]
    out[:h_half, w_half:] = x[1]
    out[h_half:, :w_half] = x[2]
    out[h_half:, w_half:] = x[3]
    return out


def local_adaptive(img):
    parts = split4(img)
    res = []
    for i in range(4):
        # หา threshold แยกของแต่ละส่วน จึงทนกับภาพที่แสงไม่เท่ากันได้
        T = otsu(parts[i])
        res.append(binarize(parts[i], T))
    out = np.zeros(img.shape, np.uint8)
    return MergeLocal(res, out)


if __name__ == "__main__":
    img = cv2.imread("./images/shade.png", 0)
    out = local_adaptive(img)
    cv2.imshow("local adaptive", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
