import cv2
import time

from config import TCP_PORT, HOST_IP, RGB_FPS
from protocol import (
    create_frame_packet,
    PIXEL_FORMAT_BGR24
)
from transport import TCPTransport

VIDEO_PATH = "video/test.mp4"


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("影片開啟失敗！")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0 or total_frames <= 0 or RGB_FPS <= 0:
        print("影片資訊或 RGB_FPS 設定錯誤！")
        cap.release()
        return

    print("影片開啟成功")
    print("影片原始 FPS:", source_fps)
    print("傳送 FPS:", RGB_FPS)
    print("目標電腦:", HOST_IP)

    transport = TCPTransport(HOST_IP, TCP_PORT)
    frame_id = 1
    sample_index = 0
    next_send_time = time.monotonic()

    try:
        print("正在連線...")
        transport.connect()
        print("連線成功，開始傳送彩色影片！")
        print("按 Ctrl + C 停止傳送")

        while True:
            # 計算這次要讀取影片的哪一幀
            target_frame = round(
                sample_index * source_fps / RGB_FPS
            )

            # 影片播完就回到開頭，繼續傳送
            if target_frame >= total_frames:
                sample_index = 0
                target_frame = 0

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()

            if not ret:
                sample_index = 0
                continue

            height, width = frame.shape[:2]

            # 彩色 BGR 畫面轉成 bytes
            payload = frame.tobytes()

            # 建立彩色封包
            packet = create_frame_packet(
                frame_id=frame_id,
                width=width,
                height=height,
                payload=payload,
                pixel_format=PIXEL_FORMAT_BGR24
            )

            # 控制傳送速度
            wait_time = next_send_time - time.monotonic()
            if wait_time > 0:
                time.sleep(wait_time)

            # 傳送封包
            transport.send(packet)

            print(
                f"已傳送彩色 Frame {frame_id}，"
                f"封包大小：{len(packet)} bytes"
            )

            frame_id += 1
            sample_index += 1
            next_send_time += 1 / RGB_FPS

    except KeyboardInterrupt:
        print("\n收到停止指令，結束傳送。")

    except (ConnectionError, OSError) as e:
        print("傳送中斷：", e)

    finally:
        cap.release()
        transport.close()
        print("傳送端已關閉。")


if __name__ == "__main__":
    main()
