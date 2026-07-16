import cv2


def otsu(img):
    # ส่ง T = 0 แล้วให้ flag OTSU หาค่าเอง ret คือ T ที่หาได้
    ret, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret, out


if __name__ == "__main__":
    img = cv2.imread("./images/document1.png", 0)
    T, out = otsu(img)
    print("threshold =", T)
    cv2.imshow("otsu", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
