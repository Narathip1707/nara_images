import cv2
import matplotlib.pyplot as plt


def histogram(img):
    # calcHist คืนค่าเป็น shape (256,1) ต้อง flatten ก่อนใช้
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist.flatten()


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    hist = histogram(img)
    plt.bar(range(256), hist)
    plt.show()
