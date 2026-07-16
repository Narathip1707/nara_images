import cv2
import numpy as np


def log_transform(img):
    rows, cols = img.shape
    # ต้องแปลงเป็น float ก่อนเสมอ ถ้าคำนวณบน uint8 ตรงๆ จะเจอ 2 ปัญหา:
    #   1) 1 + 255 วนกลับเป็น 0 -> log(0) = -inf -> ภาพออกมาดำทั้งรูป
    #   2) np.log ของ uint8 คืน float16 ทำให้ c คลาดเคลื่อน
    f = img.astype(np.float64)
    r_max = float(np.amax(f))

    # c คำนวณจากค่าสูงสุดของภาพ เพื่อ scale ผลลัพธ์กลับมาที่ 0-255
    c = 255.0 / np.log(1.0 + r_max)

    out = np.zeros((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            r = f[y, x]
            out[y, x] = int(c * np.log(1.0 + r))   # int() = ตัดเศษทิ้ง เหมือน .astype(uint8)
    return out


if __name__ == "__main__":
    img = cv2.imread("./images/dark2.png", 0)
    out = log_transform(img)
    cv2.imshow("log", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
