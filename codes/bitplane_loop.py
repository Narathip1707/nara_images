import cv2
import numpy as np


def bitplane(img, bit):
    rows, cols = img.shape
    out = np.zeros((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            r = img[y, x]
            # เลื่อนบิตที่ต้องการมาขวาสุด แล้ว and 1 จากนั้นคูณ 255 ให้มองเห็น
            out[y, x] = ((r >> bit) & 1) * 255
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    for b in range(8):
        cv2.imshow("bit " + str(b), bitplane(img, b))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
