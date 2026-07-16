import cv2


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


if __name__ == "__main__":
    img = cv2.imread("./images/lena_color_256.png")
    gray = to_gray(img)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
