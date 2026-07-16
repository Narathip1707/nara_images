import cv2


def equalize(img):
    return cv2.equalizeHist(img)


if __name__ == "__main__":
    img = cv2.imread("./images/dark2.png", 0)
    out = equalize(img)
    cv2.imshow("equalize", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
