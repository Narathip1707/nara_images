import cv2
import numpy as np


def matrix_translate(T, tx, ty):
    Ts = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [tx, ty, 1]], np.float64)
    return np.matmul(Ts, T)


def matrix_scale(T, sx, sy):
    S = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0, 0, 1]], np.float64)
    return np.matmul(S, T)


def matrix_rotate(T, theta):
    ang = theta * np.pi / 180
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


def composite(img, theta, sx, sy):
    rows, cols = img.shape
    xc, yc = int(rows / 2), int(cols / 2)  # centroid กลางภาพ
    T = np.identity(3)
    # ลำดับสำคัญมาก: เลื่อนไปจุดกลาง -> หมุน -> ย่อขยาย -> เลื่อนกลับ
    T = matrix_translate(T, xc, yc)
    T = matrix_rotate(T, theta)
    T = matrix_scale(T, sx, sy)
    T = matrix_translate(T, -xc, -yc)
    return apply_transform(img, T)


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    out = composite(img, 30, 0.8, 0.8)
    cv2.imshow("composite", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
