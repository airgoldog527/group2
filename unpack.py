import cv2
import numpy as np

from protocol import parse_frame_packet


# =========================
# 讀取 .bin
# =========================

with open("frames/frame_0001.bin", "rb") as f:
    packet = f.read()

print("讀取 .bin 成功！")
print("Packet 大小:", len(packet), "bytes")


# =========================
# 解析 Packet
# =========================

frame = parse_frame_packet(packet)

print()
print("===== Packet Parsed =====")

print("Frame ID:", frame["frame_id"])
print("Width:", frame["width"])
print("Height:", frame["height"])
print("Pixel Format:", frame["pixel_format"])
print("Payload Size:", frame["payload_size"])


# =========================
# bytes → NumPy
# =========================

received_gray = np.frombuffer(
    frame["payload"],
    dtype=np.uint8
)

received_gray = received_gray.reshape(
    frame["height"],
    frame["width"]
)

print()
print("還原後 shape:", received_gray.shape)
print("還原後 dtype:", received_gray.dtype)


# =========================
# 儲存還原後圖片
# =========================

cv2.imwrite(
    "decoded.png",
    received_gray
)

print()
print("===== 解包完成 =====")
print("產生圖片：decoded.png")


# =========================
# 顯示圖片
# =========================

cv2.imshow(
    "Decoded Frame",
    received_gray
)

cv2.waitKey(0)
cv2.destroyAllWindows()