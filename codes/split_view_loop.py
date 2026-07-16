import cv2
import numpy as np


def split4(img):
    h, w = img.shape
    h_half = h // 2
    w_half = w // 2
    # เรียง: บนซ้าย, บนขวา, ล่างซ้าย, ล่างขวา
    return [img[:h_half, :w_half], img[:h_half, w_half:],
            img[h_half:, :w_half], img[h_half:, w_half:]]


def MergeLocal(x, out):
    h, w = out.shape
    h_half = h // 2
    w_half = w // 2
    out[:h_half, :w_half] = x[0]
    out[:h_half, w_half:] = x[1]
    out[h_half:, :w_half] = x[2]
    out[h_half:, w_half:] = x[3]
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/shade.png", 0)
    parts = split4(img)
    for i in range(4):
        cv2.imshow("part " + str(i), parts[i])

    out = np.zeros(img.shape, np.uint8)
    out = MergeLocal(parts, out)
    cv2.imshow("merge", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
