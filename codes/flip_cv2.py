import cv2


def flip(img, opt):
    # กับดัก: flipCode ของ cv2 สลับกับ opt ของเรา
    # opt 0 (ซ้าย-ขวา) -> flipCode 1 , opt 1 (บน-ล่าง) -> flipCode 0
    if opt == 0:
        return cv2.flip(img, 1)
    elif opt == 1:
        return cv2.flip(img, 0)
    else:
        return cv2.flip(img, -1)


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    cv2.imshow("horizontal", flip(img, 0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
