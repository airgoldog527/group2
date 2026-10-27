import cv2

from protocol import create_frame_packet, PIXEL_FORMAT_GRAY8


# =========================
# 讀取圖片
# =========================

print("開始讀取圖片...")

image = cv2.imread("image/abcd.png")

print("圖片讀取函式執行完畢")

if image is None:
    print("圖片讀取失敗！")
    exit()

print("圖片讀取成功！")
print("原圖 shape:", image.shape)


# =========================
# 轉成灰階
# =========================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

height, width = gray.shape

print("灰階寬度:", width)
print("灰階高度:", height)


# =========================
# NumPy → bytes
# =========================

payload = gray.tobytes()

print("Payload 大小:", len(payload), "bytes")


# =========================
# 建立 Frame Packet
# =========================

packet = create_frame_packet(
    frame_id=1,
    width=width,
    height=height,
    payload=payload,
    pixel_format=PIXEL_FORMAT_GRAY8
)


# =========================
# 儲存成 .bin
# =========================

with open("frame_test.bin", "wb") as f:
    f.write(packet)

print()
print("===== 打包完成 =====")
print("產生檔案：frame_test.bin")
print("Packet 大小:", len(packet), "bytes")
