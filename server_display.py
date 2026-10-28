
import socket
import threading
import time

import cv2
import numpy as np

from config import (
    TCP_PORT,
    HEADER_SIZE,
    PIXEL_FORMAT_GRAY8,
    PIXEL_FORMAT_BGR24,
    PIXEL_FORMAT_RGB24,
)
from protocol import StreamReceiver

# 第四台電腦：同時接收彩色影片與分析文字
HOST = "0.0.0.0"
VIDEO_PORT = TCP_PORT       # 預設 5000
TEXT_PORT = TCP_PORT + 1    # 預設 5001

# 兩條連線共用的最新資料
lock = threading.Lock()
latest_frame = None
latest_result = "等待第三台傳送分析結果..."
latest_frame_id = 0
latest_result_frame_id = 0
running = True


def receive_video():
    """接收第一台傳來的 FRAME 封包。"""
    global latest_frame, latest_frame_id, running

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, VIDEO_PORT))
    server.listen(5)

    print(f"[影片] 等待第一台連線：Port {VIDEO_PORT}")

    while running:
        conn, addr = server.accept()
        print(f"[影片] 已連線：{addr}")
        receiver = StreamReceiver()

        with conn:
            while running:
                data = conn.recv(65536)
                if not data:
                    break

                receiver.feed(data)

                for packet in receiver.extract_packets():
                    if packet["type_name"] != "FRAME":
                        continue

                    width = packet["width"]
                    height = packet["height"]
                    fmt = packet["pixel_format"]
                    payload = packet["payload"]

                    if fmt == PIXEL_FORMAT_BGR24:
                        frame = np.frombuffer(
                            payload, dtype=np.uint8
                        ).reshape(height, width, 3).copy()

                    elif fmt == PIXEL_FORMAT_RGB24:
                        rgb = np.frombuffer(
                            payload, dtype=np.uint8
                        ).reshape(height, width, 3)
                        frame = cv2.cvtColor(
                            rgb, cv2.COLOR_RGB2BGR
                        )

                    elif fmt == PIXEL_FORMAT_GRAY8:
                        gray = np.frombuffer(
                            payload, dtype=np.uint8
                        ).reshape(height, width)
                        frame = cv2.cvtColor(
                            gray, cv2.COLOR_GRAY2BGR
                        )

                    else:
                        continue

                    with lock:
                        latest_frame = frame
                        latest_frame_id = packet["frame_id"]

        print("[影片] 傳送端已斷線，等待重新連線")

    server.close()


def receive_text():
    """接收第三台傳來的 TEXT 封包。"""
    global latest_result, latest_result_frame_id, running

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, TEXT_PORT))
    server.listen(5)

    print(f"[文字] 等待第三台連線：Port {TEXT_PORT}")

    while running:
        conn, addr = server.accept()
        print(f"[文字] 已連線：{addr}")
        receiver = StreamReceiver()

        with conn:
            while running:
                data = conn.recv(65536)
                if not data:
                    break

                receiver.feed(data)

                for packet in receiver.extract_packets():
                    if packet["type_name"] != "TEXT":
                        continue

                    message = packet["message"]
                    frame_id = packet["frame_id"]

                    with lock:
                        latest_result = message
                        latest_result_frame_id = frame_id

                    print(f"[分析結果][Frame {frame_id}] {message}")

        print("[文字] 傳送端已斷線，等待重新連線")

    server.close()


def main():
    global running

    # 兩個接收工作各自運行，互不阻塞
    video_thread = threading.Thread(
        target=receive_video, daemon=True
    )
    text_thread = threading.Thread(
        target=receive_text, daemon=True
    )

    video_thread.start()
    text_thread.start()

    print("[第四台] 等待影片與分析結果")
    print("按 Q 結束顯示")

    try:
        while True:
            with lock:
                frame = (
                    latest_frame.copy()
                    if latest_frame is not None
                    else None
                )
                result = latest_result
                frame_id = latest_frame_id
                result_id = latest_result_frame_id

            if frame is None:
                canvas = np.zeros((480, 800, 3), dtype=np.uint8)
                cv2.putText(
                    canvas, "Waiting for color video...",
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2
                )
            else:
                canvas = frame

            # 顯示最新分析結果
            lines = result.splitlines() or ["(空白結果)"]
            y = 30

            cv2.rectangle(
                canvas, (5, 5),
                (min(canvas.shape[1] - 5, 900),
                 min(canvas.shape[0] - 5, 40 + 30 * len(lines))),
                (0, 0, 0), -1
            )

            cv2.putText(
                canvas,
                f"Video Frame: {frame_id} | Result Frame: {result_id}",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 1, cv2.LINE_AA
            )

            for line in lines[:8]:
                y += 25
                cv2.putText(
                    canvas, line[:100],
                    (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA
                )

            cv2.imshow("Computer 4 - Color Video + Analysis", canvas)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        cv2.destroyAllWindows()
        print("[第四台] 顯示程式結束")


if __name__ == "__main__":
    main()
