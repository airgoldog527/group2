
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import TCP_PORT, HOST_IP
from protocol import StreamReceiver, PIXEL_FORMAT_GRAY8
from transport import TCPTransport


# 已經測試成功的 YOLO 模型
MODEL_PATH = "yolo26n.pt"

# 視窗標題
WINDOW_NAME = "FPGA / TCP / YOLO Overlay"

# 每次從 TCP 接收的最大資料量
RECV_SIZE = 65536


def main():
    server_transport = None
    client_transport = None

    try:
        print("=" * 50)
        print("       Continuous TCP + YOLO Overlay")
        print("=" * 50)

        # ==================================================
        # 1. 載入 YOLO 模型
        # ==================================================

        print("[YOLO] 載入模型...")

        model = YOLO(MODEL_PATH)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，停止執行。")

        print(f"[YOLO] GPU: {torch.cuda.get_device_name(0)}")
        print("[YOLO] 模型載入完成")

        # ==================================================
        # 2. 建立 TCP Server
        # ==================================================

        server_transport = TCPTransport(HOST_IP, TCP_PORT)

        print(f"[Server] 監聽 {HOST_IP}:{TCP_PORT}")
        print("[Server] 等待 Client 連線...")

        client_transport = server_transport.listen()

        print("[Server] 連線成功！")
        print("[Overlay] 按 Q 關閉視窗並停止接收")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        # ==================================================
        # 3. 初始化統計資料與串流接收器
        # ==================================================

        frame_count = 0
        process_fps = 0.0
        previous_frame_time = time.perf_counter()

        # 每條 TCP 連線只建立一次 StreamReceiver
        receiver = StreamReceiver()

        # ==================================================
        # 4. 持續接收 TCP 資料並解析影格
        # ==================================================

        stop_requested = False

        while not stop_requested:

            # TCP 是位元組串流，不保證一次收到完整封包
            data = client_transport.sock.recv(RECV_SIZE)

            if not data:
                print("[Server] Client 已斷線")
                break

            # 將收到的資料加入緩衝區
            receiver.feed(data)

            # 取出所有已完整接收、且驗證通過的封包
            for packet in receiver.extract_packets():

                # 目前這個 Server 只處理影像封包
                if packet["type_name"] != "FRAME":
                    continue

                # 取得封包資訊
                frame_id = packet["frame_id"]
                width = packet["width"]
                height = packet["height"]
                pixel_format = packet["pixel_format"]
                payload = packet["payload"]
                payload_size = packet["payload_size"]

                # ------------------------------------------
                # 4-1. 確認影像格式為 GRAY8
                # ------------------------------------------

                if pixel_format != PIXEL_FORMAT_GRAY8:
                    print(
                        f"[警告] 略過非灰階影格："
                        f"Frame={frame_id}, "
                        f"Pixel Format={pixel_format}"
                    )
                    continue

                # GRAY8 每個像素佔 1 byte
                expected_size = width * height

                if payload_size != expected_size:
                    raise ValueError(
                        f"影像大小不符："
                        f"Frame={frame_id}, "
                        f"payload={payload_size}, "
                        f"expected={expected_size}"
                    )

                # ------------------------------------------
                # 4-2. 將 Payload 還原成灰階影像
                # ------------------------------------------

                gray = np.frombuffer(
                    payload,
                    dtype=np.uint8,
                ).reshape(
                    height,
                    width,
                ).copy()

                # YOLO 使用 3 通道影像輸入。
                # 灰階資料只在接收端轉換，不改變網路傳輸內容。
                yolo_image = cv2.cvtColor(
                    gray,
                    cv2.COLOR_GRAY2BGR,
                )

                # ------------------------------------------
                # 4-3. 執行 YOLO 推論
                # ------------------------------------------

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

                # 取得包含偵測框的影像
                overlay = result.plot()

                # ------------------------------------------
                # 4-4. 計算處理 FPS
                # ------------------------------------------

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

                # ------------------------------------------
                # 4-5. 顯示影格編號、FPS、推論時間
                # ------------------------------------------

                cv2.putText(
                    overlay,
                    f"Frame: {frame_id}",
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

                # ------------------------------------------
                # 4-6. 顯示即時 YOLO Overlay
                # ------------------------------------------

                cv2.imshow(WINDOW_NAME, overlay)

                # 每 30 張影格印一次資訊
                if frame_count % 30 == 0:
                    print(
                        f"[YOLO] Frame={frame_id} | "
                        f"Detections={len(result.boxes)} | "
                        f"Inference={inference_ms:.1f} ms | "
                        f"FPS={process_fps:.1f}"
                    )

                # ------------------------------------------
                # 4-7. 按 Q 停止接收
                # ------------------------------------------

                if cv2.waitKey(1) & 0xFF in (
                    ord("q"),
                    ord("Q"),
                ):
                    print("[Overlay] 收到 Q，停止接收")
                    stop_requested = True
                    break

    except KeyboardInterrupt:
        print("\n[Server] 收到 Ctrl+C，停止執行")

    except (ConnectionError, OSError, ValueError) as e:
        print(f"\n[Server] 執行或接收錯誤：{e}")

    finally:
        cv2.destroyAllWindows()

        if client_transport is not None:
            client_transport.close()

        if server_transport is not None:
            server_transport.close()

        print("[Server] 連線已關閉")


if __name__ == "__main__":
    main()
