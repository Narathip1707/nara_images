import cv2
import numpy as np


def matrix_scale(T, sx, sy):
    S = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0, 0, 1]], np.float64)
    return np.matmul(S, T)


def apply_transform(img, T):
    rows, cols = img.shape
    out = np.zeros((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            p = np.matmul(np.array([x, y, 1], np.float64), T)
            xn = int(p[0])
            yn = int(p[1])
            if 0 <= xn < cols and 0 <= yn < rows:
                out[yn, xn] = img[y, x]
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    T = np.identity(3)
    T = matrix_scale(T, 1.5, 1.5)
    # forward map ที่ขยายภาพจะมีรูโหว่ (aliasing) เป็นเรื่องปกติของวิธีนี้
    out = apply_transform(img, T)
    cv2.imshow("scale", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
