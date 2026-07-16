import cv2


def scale(img, sx, sy):
    return cv2.resize(img, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    out = scale(img, 1.5, 1.5)
    cv2.imshow("scale", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
