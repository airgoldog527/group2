import cv2

from protocol import create_frame_packet


# 讀取圖片
image = cv2.imread("image/testpk3.jpg")

if image is None:
    raise FileNotFoundError("找不到 image/test.jpg")


# BGR → 灰階
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width = gray.shape

# 灰階影像轉成 bytes
payload = gray.tobytes()

# 建立 Frame Packet
packet = create_frame_packet(
    frame_id=5,
    width=width,
    height=height,
    payload=payload
)

# 存成二進位檔
with open("frame_005.bin", "wb") as f:
    f.write(packet)

print("封包建立完成！")
print(f"圖片尺寸：{width} × {height}")
print(f"Payload：{len(payload)} bytes")
print(f"完整封包：{len(packet)} bytes")
print("輸出檔案：frame_005.bin")