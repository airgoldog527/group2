
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import TCP_PORT, HOST_IP
from protocol import (
    StreamReceiver,
    PIXEL_FORMAT_GRAY8,
    create_text_packet,
)
from transport import TCPTransport
from client_text import YOLOTextSender


# YOLO 模型路徑
MODEL_PATH = "yolo26n.pt"

# OpenCV 視窗標題
WINDOW_NAME = "FPGA / TCP / YOLO Overlay"

# 每次從 TCP 接收的最大資料量
RECV_SIZE = 65536


def main():
    server_transport = None
    client_transport = None
    text_sender = None

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
            raise RuntimeError(
                "CUDA 不可用，請確認 NVIDIA GPU 與 PyTorch CUDA 環境。"
            )

        print(f"[YOLO] GPU: {torch.cuda.get_device_name(0)}")
        print("[YOLO] 模型載入完成")

        # ==================================================
        # 2. 連線至電腦 C 的文字接收端 TCP 5001
        # ==================================================

        print("[TEXT] 正在連線至文字接收端...")

        text_sender = YOLOTextSender()

        if not text_sender.start_connection():
            print(
                "[TEXT] 無法連線至 C 的 TCP 5001，"
                "先繼續執行 YOLO"
            )

            # 文字連線失敗時，仍然可以執行影像辨識
            try:
                text_sender.close_connection()
            except Exception:
                pass

            text_sender = None

        else:
            print("[TEXT] 文字傳送端連線成功")

        # ==================================================
        # 3. 建立 TCP Server，接收電腦 A 的影像
        # ==================================================

        server_transport = TCPTransport(HOST_IP, TCP_PORT)

        print(f"[Server] 監聽 {HOST_IP}:{TCP_PORT}")
        print("[Server] 等待 Client 連線...")

        client_transport = server_transport.listen()

        print("[Server] 影像連線成功！")
        print("[Overlay] 按 Q 關閉視窗並停止接收")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        # ==================================================
        # 4. 初始化統計資料與串流接收器
        # ==================================================

        frame_count = 0
        process_fps = 0.0
        previous_frame_time = time.perf_counter()

        # 每條 TCP 連線只建立一次 StreamReceiver
        receiver = StreamReceiver()

        # ==================================================
        # 5. 持續接收 TCP 資料並解析影格
        # ==================================================

        stop_requested = False

        while not stop_requested:

            # TCP 是位元組串流，不保證一次收到完整封包
            data = client_transport.sock.recv(RECV_SIZE)

            if not data:
                print("[Server] 影像 Client 已斷線")
                break

            # 將收到的資料加入緩衝區
            receiver.feed(data)

            # 取出完整封包
            for packet in receiver.extract_packets():

                # 只處理影像封包
                if packet["type_name"] != "FRAME":
                    continue

                frame_id = packet["frame_id"]
                width = packet["width"]
                height = packet["height"]
                pixel_format = packet["pixel_format"]
                payload = packet["payload"]
                payload_size = packet["payload_size"]

                # ------------------------------------------
                # 5-1. 確認影像格式
                # ------------------------------------------

                if pixel_format != PIXEL_FORMAT_GRAY8:
                    print(
                        f"[警告] 略過非灰階影格："
                        f"Frame={frame_id}, "
                        f"Pixel Format={pixel_format}"
                    )
                    continue

                expected_size = width * height

                if payload_size != expected_size:
                    raise ValueError(
                        f"影像大小不符："
                        f"Frame={frame_id}, "
                        f"payload={payload_size}, "
                        f"expected={expected_size}"
                    )

                # ------------------------------------------
                # 5-2. 將 Payload 還原成灰階影像
                # ------------------------------------------

                gray = np.frombuffer(
                    payload,
                    dtype=np.uint8,
                ).reshape(
                    height,
                    width,
                ).copy()

                # 將灰階影像轉成 YOLO 所需的 3 通道影像
                yolo_image = cv2.cvtColor(
                    gray,
                    cv2.COLOR_GRAY2BGR,
                )

                # ------------------------------------------
                # 5-3. 執行 YOLO GPU 推論
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

                # ------------------------------------------
                # 5-4. 整理 YOLO 偵測結果
                # ------------------------------------------

                detected_items = []

                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    class_name = model.names[cls_id]
                    confidence = float(box.conf[0].item())

                    xywh = [
                        round(float(value), 1)
                        for value in box.xywh[0].tolist()
                    ]

                    detected_items.append(
                        f"{class_name}({confidence:.2f}) "
                        f"xywh={xywh}"
                    )

                if detected_items:
                    message = " | ".join(detected_items)
                else:
                    message = "No objects detected"

                # ------------------------------------------
                # 5-5. 建立文字封包並傳送至電腦 C
                # ------------------------------------------

                if text_sender is not None:
                    try:
                        text_packet = create_text_packet(
                            message=message,
                            frame_id=frame_id,
                        )

                        text_sender.send_yolo_packet(text_packet)

                    except Exception as e:
                        print(f"[TEXT] 傳送失敗：{e}")

                        try:
                            text_sender.close_connection()
                        except Exception:
                            pass

                        # 連線失敗後停止傳送，但繼續執行 YOLO
                        text_sender = None

                # ------------------------------------------
                # 5-6. 取得包含偵測框的影像
                # ------------------------------------------

                overlay = result.plot()

                # ------------------------------------------
                # 5-7. 計算處理 FPS
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
                # 5-8. 顯示影格編號、FPS、推論時間
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
                # 5-9. 顯示即時 YOLO Overlay
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
                # 5-10. 按 Q 停止接收
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
            try:
                client_transport.close()
            except Exception as e:
                print(f"[Server] 關閉影像連線時發生錯誤：{e}")

        if server_transport is not None:
            try:
                server_transport.close()
            except Exception as e:
                print(f"[Server] 關閉 Server 時發生錯誤：{e}")

        if text_sender is not None:
            try:
                text_sender.close_connection()
            except Exception as e:
                print(f"[TEXT] 關閉文字連線時發生錯誤：{e}")

        print("[Server] 連線已關閉")


if __name__ == "__main__":
    main()
