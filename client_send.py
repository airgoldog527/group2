
import os
import time
import threading

from config import TCP_PORT, RGB_FPS, GRAY_FPS
from transport import TCPTransport


# ===== 設定 =====
COLOR_IP = "192.168.1.4"
GRAY_IP = "192.168.1.3"

COLOR_FOLDER = "frames/color"
GRAY_FOLDER = "frames/gray"


def send_packets(ip, folder, fps, label):
    if fps <= 0:
        print(f"[{label}] FPS 必須大於 0")
        return

    if not os.path.isdir(folder):
        print(f"[{label}] 找不到資料夾：{folder}")
        return

    files = sorted(
        name for name in os.listdir(folder)
        if name.lower().endswith(".bin")
        and os.path.isfile(os.path.join(folder, name))
    )

    if not files:
        print(f"[{label}] 沒有找到 .bin 檔案")
        return

    transport = TCPTransport(ip, TCP_PORT)

    try:
        transport.connect()
        print(f"[{label}] 已連線到 {ip}:{TCP_PORT}")

        next_time = time.monotonic()

        for filename in files:
            path = os.path.join(folder, filename)

            with open(path, "rb") as f:
                packet = f.read()

            if not packet:
                print(f"[{label}] 跳過空檔案：{filename}")
                continue

            delay = next_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)

            # 原封不動傳送既有封包
            transport.send(packet)

            print(
                f"[{label}] 已傳送 {filename} "
                f"（{len(packet)} bytes）"
            )

            next_time += 1 / fps

        print(f"[{label}] 全部封包傳送完成")

    except Exception as e:
        print(f"[{label}] 傳送失敗：{e}")

    finally:
        transport.close()


def main():
    color_thread = threading.Thread(
        target=send_packets,
        args=(COLOR_IP, COLOR_FOLDER, RGB_FPS, "彩色"),
    )

    gray_thread = threading.Thread(
        target=send_packets,
        args=(GRAY_IP, GRAY_FOLDER, GRAY_FPS, "灰階"),
    )

    color_thread.start()
    gray_thread.start()

    color_thread.join()
    gray_thread.join()

    print("傳送程序結束")


if __name__ == "__main__":
    main()
