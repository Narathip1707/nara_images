import cv2


def binarize(img, T):
    # threshold คืนค่า 2 ตัว ตัวแรกคือค่า T ที่ใช้จริง
    ret, out = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/document1.png", 0)
    out = binarize(img, 128)
    cv2.imshow("binarize", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
