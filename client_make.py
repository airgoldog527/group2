
import os
import time
import threading
import cv2

from config import TCP_PORT, RGB_FPS, GRAY_FPS
from protocol import (
    create_frame_packet,
    PIXEL_FORMAT_BGR24,
    PIXEL_FORMAT_GRAY8,
)
from transport import TCPTransport


# ===== 設定 =====
VIDEO_PATH = "video/test.mp4"

COLOR_IP = "192.168.1.4"
GRAY_IP = "192.168.1.3"

COLOR_FOLDER = "frames/color"
GRAY_FOLDER = "frames/gray"



def make_and_send(ip, pixel_format, folder, fps, label):
    if fps <= 0:
        print(f"[{label}] FPS 必須大於 0")
        return

    os.makedirs(folder, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"[{label}] 無法開啟影片：{VIDEO_PATH}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps <= 0:
        cap.release()
        print(f"[{label}] 無法取得影片原始 FPS")
        return

    # 每隔多少張原始影格，取樣一次
    frame_step = max(1, round(source_fps / fps))

    print(f"[{label}] 原始 FPS：{source_fps:.2f}")
    print(f"[{label}] 目標取樣 FPS：{fps}")
    print(f"[{label}] 每隔 {frame_step} 張原始影格取樣")

    transport = TCPTransport(ip, TCP_PORT)
    frame_id = 1
    source_frame_id = 0

    try:
        transport.connect()
        print(f"[{label}] 已連線到 {ip}:{TCP_PORT}")

        next_time = time.monotonic()

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            # 跳過不需要的原始影格
            if source_frame_id % frame_step != 0:
                source_frame_id += 1
                continue

            height, width = frame.shape[:2]

            if pixel_format == PIXEL_FORMAT_GRAY8:
                image = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2GRAY
                )
            else:
                image = frame

            payload = image.tobytes()

            packet = create_frame_packet(
                frame_id=frame_id,
                width=width,
                height=height,
                pixel_format=pixel_format,
                payload=payload,
            )

            filename = os.path.join(
                folder, f"frame_{frame_id:04d}.bin"
            )

            with open(filename, "wb") as f:
                f.write(packet)

            # 控制封包傳送的時間間隔
            delay = next_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)

            transport.send(packet)

            print(
                f"[{label}] 封包 {frame_id}，"
                f"原始影格 {source_frame_id}，"
                f"{len(packet)} bytes"
            )

            frame_id += 1
            source_frame_id += 1
            next_time += 1 / fps

        print(f"[{label}] 完成，共製作 {frame_id - 1} 個封包")

    except Exception as e:
        print(f"[{label}] 發生錯誤：{e}")

    finally:
        cap.release()
        transport.close()



def main():
    color_thread = threading.Thread(
        target=make_and_send,
        args=(
            COLOR_IP,
            PIXEL_FORMAT_BGR24,
            COLOR_FOLDER,
            RGB_FPS,
            "彩色",
        ),
    )

    gray_thread = threading.Thread(
        target=make_and_send,
        args=(
            GRAY_IP,
            PIXEL_FORMAT_GRAY8,
            GRAY_FOLDER,
            GRAY_FPS,
            "灰階",
        ),
    )

    color_thread.start()
    gray_thread.start()

    color_thread.join()
    gray_thread.join()

    print("彩色與灰階傳送程序結束")


if __name__ == "__main__":
    main()
