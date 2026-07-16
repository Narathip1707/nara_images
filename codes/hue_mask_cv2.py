import cv2
import numpy as np


def hue_mask(img, low, high):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    # กับดัก: H ของ OpenCV เป็น 0-179 ต้องหาร 2 จากองศาจริงก่อนเทียบ
    mask = (h >= low / 2) & (h <= high / 2)
    out = np.zeros_like(img)
    out[mask] = img[mask]
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/RGB_test.png")
    out = hue_mask(img, 0, 60)
    cv2.imshow("hue mask", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
