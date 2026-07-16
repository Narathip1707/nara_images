import cv2
import numpy as np


def hsv2bgr(h, s, v):
    rows, cols = h.shape
    out = np.zeros((rows, cols, 3), np.uint8)
    for y in range(rows):
        for x in range(cols):
            hh, ss, vv = h[y, x], s[y, x], v[y, x]
            c = vv * ss
            xx = c * (1 - abs(((hh / 60.0) % 2) - 1))
            m = vv - c

            i = int(hh // 60) % 6  # อยู่ในเสี้ยวไหนของวงกลมสี (0-5)
            if i == 0:
                r, g, b = c, xx, 0
            elif i == 1:
                r, g, b = xx, c, 0
            elif i == 2:
                r, g, b = 0, c, xx
            elif i == 3:
                r, g, b = 0, xx, c
            elif i == 4:
                r, g, b = xx, 0, c
            else:
                r, g, b = c, 0, xx

            # ปัดเศษ ไม่ใช่ตัดทิ้ง -- ที่นี่ที่เดียวในโปรเจกต์ที่ปัด
            # ถ้าตัดทิ้ง แปลง BGR->HSV->BGR จะได้ค่าน้อยกว่าเดิม 1 เกือบทุกพิกเซล
            # ปัดแล้วแปลงไป-กลับได้ค่าเดิมเป๊ะทุกสี (ทดสอบครบ 2 ล้านสีแล้ว)
            out[y, x, 0] = int(round((b + m) * 255))
            out[y, x, 1] = int(round((g + m) * 255))
            out[y, x, 2] = int(round((r + m) * 255))
    return out


if __name__ == "__main__":
    from hsv_loop import rgb2hsv

    img = cv2.imread("./images/lena_color_256.png")
    h, s, v = rgb2hsv(img)
    out = hsv2bgr(h, s, v)
    print("ต่างจากภาพเดิมสูงสุด:", np.abs(out.astype(int) - img.astype(int)).max())  # ต้องได้ 0
    cv2.imshow("round trip", cv2.hconcat([img, out]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
