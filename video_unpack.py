import cv2
import numpy as np

from protocol import (
    parse_frame_packet,
    PIXEL_FORMAT_GRAY8,
    PIXEL_FORMAT_BGR24
)


# =========================
# 彩色封包
# =========================

color_file = "frames/color/frame_0001.bin"

with open(color_file, "rb") as f:
    packet = f.read()

frame = parse_frame_packet(packet)

print("===== 彩色封包 =====")
print("Frame ID:", frame["frame_id"])
print("Width:", frame["width"])
print("Height:", frame["height"])
print("Pixel Format:", frame["pixel_format"])
print("Payload Size:", frame["payload_size"])


if frame["pixel_format"] == PIXEL_FORMAT_BGR24:

    color_image = np.frombuffer(
        frame["payload"],
        dtype=np.uint8
    )

    color_image = color_image.reshape(
        frame["height"],
        frame["width"],
        3
    )

    cv2.imwrite(
        "color_0001.png",
        color_image
    )

    print("彩色圖片已產生：color_0001.png")


# =========================
# 灰階封包
# =========================

gray_file = "frames/gray/frame_0001.bin"

with open(gray_file, "rb") as f:
    packet = f.read()

frame = parse_frame_packet(packet)

print()
print("===== 灰階封包 =====")
print("Frame ID:", frame["frame_id"])
print("Width:", frame["width"])
print("Height:", frame["height"])
print("Pixel Format:", frame["pixel_format"])
print("Payload Size:", frame["payload_size"])


if frame["pixel_format"] == PIXEL_FORMAT_GRAY8:

    gray_image = np.frombuffer(
        frame["payload"],
        dtype=np.uint8
    )

    gray_image = gray_image.reshape(
        frame["height"],
        frame["width"]
    )

    cv2.imwrite(
        "gray_0001.png",
        gray_image
    )

    print("灰階圖片已產生：gray_0001.png")