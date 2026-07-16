import cv2
import numpy as np


def hsv2bgr(hsv_img):
    # .astype('float32') สำคัญมาก -- โหมด float32 ของ cv2 รับ H เป็น 0-360 และ S,V เป็น 0-1
    # ซึ่งตรงกับสูตรที่อาจารย์สอนพอดี ถ้าส่ง uint8 เข้าไปมันจะตีความ H เป็น 0-179 แทน
    return (cv2.cvtColor(hsv_img.astype("float32"), cv2.COLOR_HSV2BGR) * 255).astype("uint8")


# หมายเหตุ: นี่คือโค้ดที่เราเขียนตอบไฟนอลข้อ 2 (image_tool.py) และมันใช้ได้
# แต่ทางลัดนี้ "ไม่ตรง" กับสูตรมือใน hsv2bgr_loop.py -- ต่างกัน +-1 บนพิกเซลราว 27%
# เพราะ cv2 ใช้ตาราง sector กับ cvRound ภายในของมันเอง
# แปลว่า for-loop เขียนตามให้ตรงเป๊ะไม่ได้เลย ไม่ว่าจะปัดหรือตัดทิ้ง float32 หรือ float64
# แอปนี้เลยใช้สูตรมือเป็นตัวคำนวณจริง เพื่อให้โค้ดที่โชว์ = โค้ดที่รัน
if __name__ == "__main__":
    img = cv2.imread("./images/lena_color_256.png")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[:, :, 0] *= 2.0          # H ของ cv2 เป็น 0-179 -> คูณ 2 กลับเป็นองศา
    hsv[:, :, 1] /= 255.0        # S,V ของ cv2 เป็น 0-255 -> หาร 255 กลับเป็น 0-1
    hsv[:, :, 2] /= 255.0
    out = hsv2bgr(hsv)
    print("ต่างจากภาพเดิมสูงสุด:", np.abs(out.astype(int) - img.astype(int)).max())
    cv2.imshow("round trip", cv2.hconcat([img, out]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
