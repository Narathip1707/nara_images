import cv2


def split_channel(img):
    # cv2.split คืนตามลำดับ B, G, R
    b, g, r = cv2.split(img)
    return b, g, r


if __name__ == "__main__":
    img = cv2.imread("./images/lena_color_256.png")
    b, g, r = split_channel(img)
    out = cv2.vconcat([b, g, r])
    cv2.imshow("b g r", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
