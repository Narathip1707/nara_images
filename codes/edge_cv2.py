import cv2
import numpy as np


def edge(img):
    # ใช้ CV_64F ไม่ใช่ uint8 ไม่งั้นขอบขาว->ดำ (ค่าติดลบ) จะหายไป
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    g = cv2.magnitude(sx, sy)
    return np.clip(g, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    cv2.imshow("sobel", edge(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
