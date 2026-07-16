import cv2


def bitplane(img, bit):
    return ((img >> bit) & 1) * 255


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    cv2.imshow("bit 7", bitplane(img, 7))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
