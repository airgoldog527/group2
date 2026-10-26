
import struct
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import HEADER_FORMAT, HEADER_SIZE, CRC_SIZE, TCP_PORT
from protocol import parse_frame_packet
from transport import TCPTransport


# 你的電腦區域網路 IP
HOST = "192.168.1.3"

# 已經測試成功的 YOLO 模型
MODEL_PATH = "yolo26n.pt"

# 視窗標題
WINDOW_NAME = "FPGA / TCP / YOLO Overlay"


def main():
    server_transport = None
    client_transport = None

    try:
        print("=" * 50)
        print("       Continuous TCP + YOLO Overlay")
        print("=" * 50)

        # 1. 載入 YOLO，確認 GPU 可用
        print("[YOLO] 載入模型...")
        model = YOLO(MODEL_PATH)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，停止執行。")

        print(f"[YOLO] GPU: {torch.cuda.get_device_name(0)}")
        print("[YOLO] 模型載入完成")

        # 2. 建立 TCP 伺服器並等待朋友連線
        server_transport = TCPTransport(HOST, TCP_PORT)

        print(f"[Server] 監聽 {HOST}:{TCP_PORT}")
        print("[Server] 等待 Client 連線...")

        client_transport = server_transport.listen()

        print("[Server] 連線成功！")
        print("[Overlay] 按 Q 關閉視窗並停止接收")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        frame_count = 0
        process_fps = 0.0
        previous_frame_time = time.perf_counter()

        # 3. 持續接收影格，直到中斷或按 Q
        while True:
            # TCP 是位元組串流，必須先收完整 Header
            header = client_transport.receive_exactly(HEADER_SIZE)

            (
                magic,
                version,
                packet_type,
                frame_id,
                width,
                height,
                pixel_format,
                payload_size,
            ) = struct.unpack(HEADER_FORMAT, header)

            # 再依照 Header 宣告的長度收完 Payload + CRC
            remaining = client_transport.receive_exactly(
                payload_size + CRC_SIZE
            )

            packet = header + remaining

            # 驗證封包內容及 CRC
            frame = parse_frame_packet(packet)

            # GRAY8 每個像素必須剛好佔 1 byte
            expected_size = frame["width"] * frame["height"]

            if frame["payload_size"] != expected_size:
                raise ValueError(
                    f"影像大小不符："
                    f"payload={frame['payload_size']}, "
                    f"expected={expected_size}"
                )

            # 將 Payload 還原成灰階影像
            gray = np.frombuffer(
                frame["payload"],
                dtype=np.uint8,
            ).reshape(
                frame["height"],
                frame["width"],
            )

            # YOLO 模型需要 3 通道輸入。
            # 這裡只在接收端轉換，不會改變網路傳輸的灰階資料。
            yolo_image = cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2RGB,
            )

            # 執行 YOLO 推論
            inference_start = time.perf_counter()

            results = model.predict(
                source=yolo_image,
                device=0,
                verbose=False,
            )

            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000

            result = results[0]

            # 取得畫好偵測框的影像
            overlay = result.plot()

            # 計算實際處理 FPS
            current_time = time.perf_counter()
            elapsed = current_time - previous_frame_time
            previous_frame_time = current_time

            if elapsed > 0:
                current_fps = 1.0 / elapsed

                if frame_count == 0:
                    process_fps = current_fps
                else:
                    process_fps = (
                        0.9 * process_fps
                        + 0.1 * current_fps
                    )

            frame_count += 1

            # 顯示影格編號、FPS、推論時間
            cv2.putText(
                overlay,
                f"Frame: {frame['frame_id']}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                overlay,
                f"Process FPS: {process_fps:.1f}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                overlay,
                f"Inference: {inference_ms:.1f} ms",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # 顯示即時 Overlay
            cv2.imshow(WINDOW_NAME, overlay)

            # 每 30 張印一次資訊，避免終端機被洗版
            if frame_count % 30 == 0:
                print(
                    f"[YOLO] Frame={frame['frame_id']} | "
                    f"Detections={len(result.boxes)} | "
                    f"Inference={inference_ms:.1f} ms | "
                    f"FPS={process_fps:.1f}"
                )

            # 按 Q 停止
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Overlay] 收到 Q，停止接收")
                break

    except KeyboardInterrupt:
        print("\n[Server] 收到 Ctrl+C，停止執行")

    finally:
        cv2.destroyAllWindows()

        if client_transport is not None:
            client_transport.close()

        if server_transport is not None:
            server_transport.close()

        print("[Server] 連線已關閉")


if __name__ == "__main__":
    main()
