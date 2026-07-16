import cv2


def local_adaptive(img, block=51, c=10):
    # block ต้องเป็นเลขคี่ , c คือค่าที่ลบออกจากค่าเฉลี่ยของหน้าต่าง
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, c)


if __name__ == "__main__":
    img = cv2.imread("./images/shade.png", 0)
    out = local_adaptive(img, 51, 10)
    cv2.imshow("local adaptive", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
