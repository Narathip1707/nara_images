import cv2
import numpy as np
import math


def edge(img, k):
    # k = 1 -> Prewitt , k = 2 -> Sobel (ต่างกันแค่น้ำหนักแถวกลาง)
    gx = [[-1, 0, 1],
          [-k, 0, k],
          [-1, 0, 1]]
    gy = [[-1, -k, -1],
          [0, 0, 0],
          [1, k, 1]]
    rows, cols = img.shape
    out = np.zeros((rows, cols), np.uint8)
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            sx = 0.0
            sy = 0.0
            for i in range(3):
                for j in range(3):
                    p = int(img[y - 1 + i, x - 1 + j])
                    sx += gx[i][j] * p
                    sy += gy[i][j] * p
            g = math.sqrt(sx ** 2 + sy ** 2)
            if g > 255:
                g = 255   # ต้อง clamp ไม่งั้น uint8 จะวนกลับเป็นค่าต่ำ
            out[y, x] = int(g)
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    cv2.imshow("prewitt", edge(img, 1))
    cv2.imshow("sobel", edge(img, 2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
