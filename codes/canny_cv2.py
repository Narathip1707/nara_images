import cv2


def canny(img, low=50, high=150):
    return cv2.Canny(img, low, high)


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    out = canny(img, 50, 150)
    cv2.imshow("canny", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
