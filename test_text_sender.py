
import os
import time
import cv2

from config import TCP_PORT, GRAY_FPS
from protocol import create_frame_packet, PIXEL_FORMAT_GRAY8
from transport import TCPTransport


# ===== 模擬電腦 A 的設定 =====
VIDEO_PATH = "video/test.mp4"

# 如果模擬程式和電腦 B 在同一台電腦，使用 127.0.0.1。
# 如果電腦 B 是另一台電腦，改成電腦 B 的實際 IP。
TARGET_B_IP = "127.0.0.1"

TARGET_PORT = TCP_PORT


def main():
    if GRAY_FPS <= 0:
        print("[錯誤] GRAY_FPS 必須大於 0")
        return

    if not os.path.isfile(VIDEO_PATH):
        print(f"[錯誤] 找不到影片：{VIDEO_PATH}")
        print("請確認 video 資料夾中有 test.mp4。")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"[錯誤] 無法開啟影片：{VIDEO_PATH}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps <= 0:
        print("[錯誤] 無法取得影片原始 FPS")
        cap.release()
        return

    frame_step = max(1, round(source_fps / GRAY_FPS))

    print(f"[影片] {VIDEO_PATH}")
    print(f"[影片] 原始 FPS：{source_fps:.2f}")
    print(f"[模擬 A] 傳送 FPS：{GRAY_FPS}")
    print(f"[模擬 A] 每隔 {frame_step} 張原始影格取樣")
    print(f"[模擬 A] 目標：{TARGET_B_IP}:{TARGET_PORT}")

    transport = TCPTransport(
        host=TARGET_B_IP,
        port=TARGET_PORT,
        timeout=5.0,
    )

    frame_id = 1
    source_frame_id = 0
    next_time = time.monotonic()

    try:
        transport.connect()
        print("[模擬 A] 已連線到電腦 B，開始傳送影像。")

        while True:
            ok, frame = cap.read()

            if not ok:
                print("[模擬 A] 影片播放完畢。")
                break

            # 按照目標 FPS 取樣。
            if source_frame_id % frame_step != 0:
                source_frame_id += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape

            packet = create_frame_packet(
                frame_id=frame_id,
                width=width,
                height=height,
                pixel_format=PIXEL_FORMAT_GRAY8,
                payload=gray.tobytes(),
            )

            # 控制傳送時間間隔。
            delay = next_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)

            transport.send(packet)

            print(
                f"[模擬 A] 已傳送 frame_id={frame_id}，"
                f"原始影格={source_frame_id}，"
                f"尺寸={width}x{height}，"
                f"封包={len(packet)} bytes"
            )

            frame_id += 1
            source_frame_id += 1
            next_time += 1 / GRAY_FPS

    except ConnectionRefusedError:
        print(
            "[錯誤] 電腦 B 拒絕連線。"
            "請先啟動 B 的 YOLO 伺服器，"
            "並確認它正在監聽 Port 5000。"
        )

    except Exception as e:
        print(f"[錯誤] {type(e).__name__}: {e}")

    finally:
        cap.release()
        transport.close()
        print("[模擬 A] 傳送程序結束。")


if __name__ == "__main__":
    main()


