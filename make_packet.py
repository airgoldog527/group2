
import cv2

from protocol import create_frame_packet, PIXEL_FORMAT_BGR24


# 讀取圖片
image = cv2.imread("image/testpk3.jpg")

if image is None:
    raise FileNotFoundError("找不到 image/testpk3.jpg")


# 保留 BGR 彩色影像
height, width, channels = image.shape

# 彩色影像轉成 bytes
payload = image.tobytes()


# 建立 Frame Packet
packet = create_frame_packet(
    frame_id=1,
    width=width,
    height=height,
    payload=payload,
    pixel_format=PIXEL_FORMAT_BGR24
)


# 存成二進位檔
with open("frame_001.bin", "wb") as f:
    f.write(packet)

print("封包建立完成！")
print(f"圖片尺寸：{width} × {height}")
print(f"色彩通道：{channels}")
print(f"Payload：{len(payload)} bytes")
print(f"完整封包：{len(packet)} bytes")
print("輸出檔案：frame_001.bin")
