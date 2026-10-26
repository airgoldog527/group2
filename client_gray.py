```python
import cv2
import time

from config import TCP_PORT, HOST_IP, GRAY_FPS
from protocol import create_frame_packet, PIXEL_FORMAT_GRAY8
from transport import TCPTransport


# 原始影片路徑
VIDEO_PATH = "video/test.mp4"


def main():
    # 開啟影片
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"影片開啟失敗！請確認 {VIDEO_PATH} 是否存在。")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0 or total_frames <= 0 or GRAY_FPS <= 0:
        print("影片資訊或 GRAY_FPS 設定錯誤！")
        cap.release()
        return

    print("===== 灰階影片即時轉換與傳送 =====")
    print("影片路徑：", VIDEO_PATH)
    print("影片原始 FPS：", source_fps)
    print("灰階傳送 FPS：", GRAY_FPS)
    print("接收端 IP：", HOST_IP)
    print("TCP Port：", TCP_PORT)
    print("按 Ctrl + C 停止傳送")
    print("================================")

    transport = TCPTransport(HOST_IP, TCP_PORT)

    frame_id = 1
    sample_index = 0
    next_send_time = time.monotonic()

    try:
        # 連線到接收端
        print("正在連接接收端...")
        transport.connect()
        print("連線成功！開始傳送灰階影片。")

        while True:
            # 依照指定的傳送 FPS 選擇影片影格
            target_frame = round(
                sample_index * source_fps / GRAY_FPS
            )

            # 影片播完後回到開頭
            if target_frame >= total_frames:
                sample_index = 0
                target_frame = 0
                next_send_time = time.monotonic()

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()

            if not ret:
                print("讀取影片影格失敗，停止傳送。")
                break

            # 1. 彩色畫面轉成灰階
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            height, width = gray.shape

            # 2. 灰階影像轉成 bytes
            payload = gray.tobytes()

            # 3. 使用新版 protocol.py 建立封包
            packet = create_frame_packet(
                frame_id=frame_id,
                width=width,
                height=height,
                payload=payload,
                pixel_format=PIXEL_FORMAT_GRAY8,
            )

            # 4. 控制傳送速度
            wait_time = next_send_time - time.monotonic()
            if wait_time > 0:
                time.sleep(wait_time)

            # 5. 直接傳送完整封包，不額外加 4-byte 長度前綴
            transport.send(packet)

            print(
                f"已傳送灰階 Frame {frame_id}，"
                f"尺寸：{width}x{height}，"
                f"封包大小：{len(packet)} bytes"
            )

            frame_id += 1
            sample_index += 1
            next_send_time += 1 / GRAY_FPS

    except KeyboardInterrupt:
        print("\n收到停止指令，結束傳送。")

    except (ConnectionError, OSError) as e:
        print("傳送中斷：", e)

    finally:
        cap.release()
        transport.close()
        cv2.destroyAllWindows()
        print("傳送端已關閉。")


if __name__ == "__main__":
    main()
```
