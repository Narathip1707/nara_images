import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram(img):
    rows, cols = img.shape
    hist = np.zeros(256, np.int64)  # 256 bin สำหรับค่าสีเทา 0-255
    for y in range(rows):
        for x in range(cols):
            hist[img[y, x]] += 1
    return hist


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    hist = histogram(img)
    plt.bar(range(256), hist)
    plt.show()
