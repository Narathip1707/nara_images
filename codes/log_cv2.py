import cv2
import numpy as np


def log_transform(img):
    c = 255 / np.log(1 + np.amax(img))
    out = c * np.log(1 + img.astype(np.float64))
    return out.astype(np.uint8)


if __name__ == "__main__":
    img = cv2.imread("./images/dark_image.png", 0)
    out = log_transform(img)
    cv2.imshow("log", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
