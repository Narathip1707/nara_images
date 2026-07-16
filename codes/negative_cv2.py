import cv2


def negative(img):
    return 255 - img


if __name__ == "__main__":
    img = cv2.imread("./images/breast.bmp", 0)
    out = negative(img)
    cv2.imshow("negative", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
