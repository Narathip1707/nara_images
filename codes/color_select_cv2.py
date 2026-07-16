import cv2
import numpy as np


def color_select(img, bands, s_min=0.25, v_min=0.15, drop="gray"):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rows, cols = img.shape[0], img.shape[1]

    mask = np.zeros((rows, cols), np.uint8)
    for low, high in bands:
        # cv2.inRange รับ S/V floor มาให้ในตัว (H หาร 2, S/V คูณ 255)
        lo_sv = (int(s_min * 255), int(v_min * 255))
        if low <= high:
            m = cv2.inRange(hsv, (low // 2, lo_sv[0], lo_sv[1]), (high // 2, 255, 255))
        else:
            # ช่วงวนผ่าน 0 องศา (สีแดง) ต้องยิง 2 ครั้งแล้ว or กัน
            m1 = cv2.inRange(hsv, (low // 2, lo_sv[0], lo_sv[1]), (179, 255, 255))
            m2 = cv2.inRange(hsv, (0, lo_sv[0], lo_sv[1]), (high // 2, 255, 255))
            m = cv2.bitwise_or(m1, m2)
        mask = cv2.bitwise_or(mask, m)

    if drop == "gray":
        base = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif drop == "white":
        base = np.full_like(img, 255)
    else:
        base = np.zeros_like(img)

    keep = cv2.bitwise_and(img, img, mask=mask)
    rest = cv2.bitwise_and(base, base, mask=cv2.bitwise_not(mask))
    return cv2.add(keep, rest)


# หมายเหตุ: เวอร์ชันนี้ไม่ตรงกับ color_select_loop.py เป๊ะๆ 2 จุด
#   1) cv2 เก็บ H เป็น 0-179 (ปัดเป็นจำนวนเต็มแล้วหาร 2) ทำให้เทียบได้แค่องศาคู่
#      -> พิกเซลที่ hue อยู่ตรงขอบช่วงพอดีจะเข้า/ไม่เข้า mask ต่างจากสูตรมือ
#   2) cv2.cvtColor(BGR2GRAY) ใช้น้ำหนักคนละชุดกับ 0.2989/0.5870/0.1140 ที่อาจารย์สอน
# อาจารย์ไม่เคยใช้ COLOR_BGR2HSV เลยสักที่ในสไลด์หรือ week1-11 -> ตอบข้อสอบใช้แบบ loop
if __name__ == "__main__":
    img = cv2.imread("./images/lena_color_256.png")
    out = color_select(img, [(200, 260), (20, 60)])
    cv2.imshow("color select", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
