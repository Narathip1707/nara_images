import cv2


def median_filter(img, k):
    # k ต้องเป็นเลขคี่
    return cv2.medianBlur(img, k)


if __name__ == "__main__":
    img = cv2.imread("./images/lena_salt_512.png", 0)
    out = median_filter(img, 3)
    cv2.imshow("median", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
