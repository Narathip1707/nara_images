import cv2


def transpose(img):
    return cv2.transpose(img)


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    out = transpose(img)
    cv2.imshow("transpose", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
