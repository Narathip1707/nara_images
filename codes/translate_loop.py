import cv2
import numpy as np


def matrix_translate(T, tx, ty):
    # ใช้แบบ row-vector [x,y,1] @ T ดังนั้น tx,ty อยู่แถวล่าง T[2,0], T[2,1]
    Ts = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [tx, ty, 1]], np.float64)
    return np.matmul(Ts, T)


def apply_transform(img, T):
    rows, cols = img.shape
    out = np.zeros((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            p = np.matmul(np.array([x, y, 1], np.float64), T)
            xn = int(p[0])   # ตัดทศนิยมทิ้ง ไม่ใช่ปัดเศษ
            yn = int(p[1])
            if 0 <= xn < cols and 0 <= yn < rows:
                out[yn, xn] = img[y, x]
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    T = np.identity(3)
    T = matrix_translate(T, 50, 30)
    out = apply_transform(img, T)
    cv2.imshow("translate", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
