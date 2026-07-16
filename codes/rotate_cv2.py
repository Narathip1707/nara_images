import cv2


def rotate(img, theta):
    rows, cols = img.shape
    center = (cols / 2, rows / 2)  # getRotationMatrix2D รับ (x, y)
    M = cv2.getRotationMatrix2D(center, theta, 1.0)
    return cv2.warpAffine(img, M, (cols, rows))


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    out = rotate(img, 30)
    cv2.imshow("rotate", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
