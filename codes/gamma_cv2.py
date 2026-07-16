import cv2
import numpy as np


def gamma_transform(img, gamma):
    c = 255.0
    img_norm = img / np.amax(img)
    out = c * (img_norm ** gamma)
    return out.astype(np.uint8)


if __name__ == "__main__":
    img = cv2.imread("./images/Xray_share.jpg", 0)
    out = gamma_transform(img, 0.4)
    cv2.imshow("gamma", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
