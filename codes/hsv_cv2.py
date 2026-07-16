import cv2


def rgb2hsv(img):
    # กับดัก: OpenCV เก็บ H เป็น 0-179 (ไม่ใช่ 0-359) ส่วน S,V เป็น 0-255
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return h, s, v


if __name__ == "__main__":
    img = cv2.imread("./images/lena_color_256.png")
    h, s, v = rgb2hsv(img)
    cv2.imshow("H", h)
    cv2.imshow("S", s)
    cv2.imshow("V", v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
