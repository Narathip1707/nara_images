import cv2


def composite(img, theta, scale):
    rows, cols = img.shape
    center = (cols / 2, rows / 2)
    # getRotationMatrix2D รวมหมุน + ย่อขยายรอบจุดกลาง ให้ในขั้นตอนเดียว
    M = cv2.getRotationMatrix2D(center, theta, scale)
    return cv2.warpAffine(img, M, (cols, rows))


if __name__ == "__main__":
    img = cv2.imread("./images/cameraman.png", 0)
    out = composite(img, 30, 0.8)
    cv2.imshow("composite", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
