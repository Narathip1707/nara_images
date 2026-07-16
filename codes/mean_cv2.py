import cv2


def mean_filter(img, k):
    return cv2.blur(img, (k, k))


if __name__ == "__main__":
    img = cv2.imread("./images/lena_noise_512.png", 0)
    out = mean_filter(img, 3)
    cv2.imshow("mean", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
