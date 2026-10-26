import socket
import struct
import time
from datetime import datetime
import cv2
import numpy as np

from config import (
    TCP_PORT,
    HEADER_FORMAT,
    HEADER_SIZE,
    CRC_SIZE,
    PIXEL_FORMAT_GRAY8,
    PIXEL_FORMAT_BGR24,
    PIXEL_FORMAT_RGB24,
)
from protocol import parse_frame_packet

HOST = "0.0.0.0"

def recv_exact(sock, n_bytes):
    """確保完整接收 n 個位元組"""
    data = bytearray()
    while len(data) < n_bytes:
        chunk = sock.recv(n_bytes - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)

def draw_ui_overlay(img, fps):
    """最穩定的文字疊加方式"""
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fps_str = f"FPS: {fps:.1f}"

    # 除錯印出：確認有跑進來，並確認當前影像形狀與型態
    print(f"[DEBUG UI] 正在繪製 UI -> 影像形狀: {img.shape}, dtype: {img.dtype}, FPS: {fps:.1f}")

    # 1. 直接畫一個實心黑色底塊 (左上角 10,10 到 450,90)
    cv2.rectangle(img, (10, 10), (450, 90), (0, 0, 0), -1)

    # 2. 畫亮綠色 FPS 文字
    cv2.putText(
        img,
        fps_str,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # 3. 畫亮黃色時間文字
    cv2.putText(
        img,
        current_time_str,
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img

def main():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, TCP_PORT))
    server_sock.listen(1)

    print(f"[*] Display Server 啟動成功，監聽連接埠 (Port): {TCP_PORT}")
    print("[*] 等待傳送端 (192.168.1.2) 連線...")

    while True:
        try:
            client_sock, client_addr = server_sock.accept()
            print(f"[+] 收到連線來自：{client_addr}")

            # FPS 計算變數
            prev_time = time.time()
            fps = 0.0

            while True:
                # 1. 讀取 Header
                header_data = recv_exact(client_sock, HEADER_SIZE)
                if not header_data:
                    print("[-] 對方中斷連線")
                    break

                # 2. 解析 Header 取出 payload_size
                header_fields = struct.unpack(HEADER_FORMAT, header_data)
                payload_size = header_fields[7]

                # 3. 讀取 Payload 與 CRC
                tail_data = recv_exact(client_sock, payload_size + CRC_SIZE)
                if not tail_data:
                    print("[-] 封包接收不完整")
                    break

                # 4. 解析封包（防護非 FRAME 封包中斷連線）
                full_packet = header_data + tail_data
                try:
                    frame = parse_frame_packet(full_packet)
                except ValueError as ve:
                    print(f"[*] 收到非影像封包已忽略: {ve}")
                    continue
                except Exception as e:
                    print(f"[!] 封包解析失敗: {e}")
                    continue

                if not frame:
                    continue

                # 注意：以下這 4 行必須與 try/except 同層級，且在判斷 fmt 之前執行
                width = frame["width"]
                height = frame["height"]
                fmt = frame["pixel_format"]
                payload = frame["payload"]

                # 5. 還原為影像 Array
                if fmt == PIXEL_FORMAT_BGR24:
                    img = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3)).copy()
                elif fmt == PIXEL_FORMAT_RGB24:
                    img_rgb = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3))
                    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                elif fmt == PIXEL_FORMAT_GRAY8:
                    img_gray = np.frombuffer(payload, dtype=np.uint8).reshape((height, width))
                    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                else:
                    print(f"[!] 未支援的像素格式: {fmt}")
                    continue

                # ★關鍵：強制複製一份獨立且連續的記憶體，確保可被寫入
                img = np.ascontiguousarray(img.copy())

                # 6. 計算即時 FPS
                curr_time = time.time()
                time_diff = curr_time - prev_time
                if time_diff > 0:
                    current_instant_fps = 1.0 / time_diff
                    fps = 0.9 * fps + 0.1 * current_instant_fps if fps > 0 else current_instant_fps
                prev_time = curr_time

                # 7. 疊加 UI
                img = draw_ui_overlay(img, fps)

                # 8. 顯示視窗
                cv2.imshow("Live Video Stream", img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[*] 結束顯示")
                    return

        except Exception as err:
            import traceback
            print(f"[!] 發生異常：{err}")
            traceback.print_exc()
        finally:
            if 'client_sock' in locals():
                client_sock.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()