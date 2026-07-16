import cv2
import numpy as np


def gamma_transform(img, gamma):
    rows, cols = img.shape
    c = 255.0
    mx = np.amax(img)
    out = np.zeros((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            # ต้อง normalize เป็น [0,1] ก่อนยกกำลัง ไม่งั้นค่าทะลุ 255
            r = img[y, x] / mx
            out[y, x] = c * (r ** gamma)
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/Xray_share.jpg", 0)
    out = gamma_transform(img, 0.4)  # gamma < 1 ภาพสว่างขึ้น
    cv2.imshow("gamma", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
