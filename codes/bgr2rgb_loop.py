import cv2
import numpy as np


def bgr2rgb(img):
    rows, cols = img.shape[0], img.shape[1]
    out = np.zeros((rows, cols, 3), np.uint8)
    for y in range(rows):
        for x in range(cols):
            # สลับ channel 0 (B) กับ channel 2 (R) ส่วน G อยู่ที่เดิม
            out[y, x, 0] = img[y, x, 2]
            out[y, x, 1] = img[y, x, 1]
            out[y, x, 2] = img[y, x, 0]
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/lena_color_256.png")
    rgb = bgr2rgb(img)
    cv2.imshow("rgb", rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
