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


# ==============================
# 設定
# ==============================

CAMERA_INDEX = 0

COLOR_IP = "192.168.1.4"
GRAY_IP = "192.168.1.3"

COLOR_FOLDER = "frames/color"
GRAY_FOLDER = "frames/gray"


# ==============================
# 攝像頭擷取與封包傳送
# ==============================

def camera_sender(ip, pixel_format, folder, fps, label, stop_event):

    if fps <= 0:
        print(f"[{label}] FPS 必須大於 0")
        return

    os.makedirs(folder, exist_ok=True)

    transport = TCPTransport(ip, TCP_PORT)
    frame_id = 1

    try:
        transport.connect()
        print(f"[{label}] 已連線到 {ip}:{TCP_PORT}")

        next_time = time.monotonic()

        while not stop_event.is_set():

            # 從攝像頭讀取即時影格
            with camera_lock:
                if latest_frame[0] is None:
                    continue
                frame = latest_frame[0].copy()

            height, width = frame.shape[:2]

            # 彩色或灰階處理
            if pixel_format == PIXEL_FORMAT_GRAY8:
                image = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2GRAY
                )
            else:
                image = frame

            payload = image.tobytes()

            # 製作封包
            packet = create_frame_packet(
                frame_id=frame_id,
                width=width,
                height=height,
                pixel_format=pixel_format,
                payload=payload,
            )

            # 儲存封包
            filename = os.path.join(
                folder, f"frame_{frame_id:04d}.bin"
            )

            with open(filename, "wb") as f:
                f.write(packet)

            # 控制傳送頻率
            delay = next_time - time.monotonic()

            if delay > 0:
                time.sleep(delay)

            if stop_event.is_set():
                break

            transport.send(packet)

            print(
                f"[{label}] frame={frame_id}, "
                f"{len(packet)} bytes"
            )

            frame_id += 1
            next_time += 1 / fps

    except Exception as e:
        print(f"[{label}] 發生錯誤：{e}")
        stop_event.set()

    finally:
        transport.close()
        print(f"[{label}] 傳送結束")


# ==============================
# 主程式
# ==============================

latest_frame = [None]
camera_lock = threading.Lock()


def main():

    stop_event = threading.Event()

    # 開啟筆電攝像頭
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("無法開啟攝像頭，請確認攝像頭沒有被其他程式占用。")
        return

    # 設定攝像頭解析度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("攝像頭已開啟。按 Q 結束。")

    color_thread = threading.Thread(
        target=camera_sender,
        args=(
            COLOR_IP,
            PIXEL_FORMAT_BGR24,
            COLOR_FOLDER,
            RGB_FPS,
            "彩色",
            stop_event,
        ),
    )

    gray_thread = threading.Thread(
        target=camera_sender,
        args=(
            GRAY_IP,
            PIXEL_FORMAT_GRAY8,
            GRAY_FOLDER,
            GRAY_FPS,
            "灰階",
            stop_event,
        ),
    )

    color_thread.start()
    gray_thread.start()

    try:
        while not stop_event.is_set():

            ok, frame = cap.read()

            if not ok:
                print("無法讀取攝像頭影格")
                break

            with camera_lock:
                latest_frame[0] = frame.copy()

            # 顯示攝像頭畫面
            cv2.imshow("Laptop Camera", frame)

            # 按 Q 結束
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()

        color_thread.join()
        gray_thread.join()

        print("攝像頭與傳送程序已結束。")


if __name__ == "__main__":
    main()