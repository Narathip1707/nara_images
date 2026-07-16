import cv2
import numpy as np


def matrix_rotate(T, theta):
    ang = theta * np.pi / 180  # np.cos/np.sin รับเรเดียน ต้องแปลงองศาก่อน
    R = np.array([[np.cos(ang), np.sin(ang), 0],
                  [-np.sin(ang), np.cos(ang), 0],
                  [0, 0, 1]], np.float64)
    return np.matmul(R, T)


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
    T = matrix_rotate(T, 30)
    # หมุนรอบจุด (0,0) ภาพจะหลุดขอบ ถ้าอยากหมุนรอบกลางภาพดู composite
    out = apply_transform(img, T)
    cv2.imshow("rotate", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
