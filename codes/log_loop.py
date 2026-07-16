import cv2
import numpy as np


def log_transform(img):
    rows, cols = img.shape
    # c คำนวณจากค่าสูงสุดของภาพ เพื่อ scale ผลลัพธ์กลับมาที่ 0-255
    c = 255 / np.log(1 + np.amax(img))
    out = np.zeros((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            r = img[y, x]
            out[y, x] = c * np.log(1 + r)
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/dark_image.png", 0)
    out = log_transform(img)
    cv2.imshow("log", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
