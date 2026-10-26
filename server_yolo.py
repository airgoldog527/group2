
import struct
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import (
    HEADER_FORMAT,
    HEADER_SIZE,
    CRC_SIZE,
    TCP_PORT,
)
from protocol import parse_frame_packet
from transport import TCPTransport


# ============================================================
# Server Settings
# ============================================================

HOST = "192.168.1.5"
MODEL_PATH = "yolo26n.pt"
OUTPUT_PATH = "yolo_tcp_result.jpg"


# ============================================================
# Server
# ============================================================

def main():
    server_transport = None
    client_transport = None

    try:
        print("========================================")
        print("       TCP + YOLO Frame Server")
        print("========================================")

        # 1. 載入模型
        print("[YOLO] 載入模型...")

        model = YOLO(MODEL_PATH)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，停止執行。")

        print(f"[YOLO] GPU: {torch.cuda.get_device_name(0)}")
        print("[YOLO] 模型載入完成")

        # 2. 建立 TCP Server
        server_transport = TCPTransport(HOST, TCP_PORT)

        print(f"[Server] 監聽 {HOST}:{TCP_PORT}")
        print("[Server] 等待 Client 連線...")

        client_transport = server_transport.listen()

        # 3. 接收 Header
        print("[Server] 等待 Header...")

        header = client_transport.receive_exactly(
            HEADER_SIZE
        )

        (
            magic,
            version,
            packet_type,
            frame_id,
            width,
            height,
            pixel_format,
            payload_size,
        ) = struct.unpack(
            HEADER_FORMAT,
            header,
        )

        print()
        print("[Server] 收到 Header")
        print(f"  Frame ID     : {frame_id}")
        print(f"  Width        : {width}")
        print(f"  Height       : {height}")
        print(f"  Payload Size : {payload_size}")

        # 4. 接收 Payload + CRC
        print("[Server] 等待 Payload + CRC...")

        remaining = client_transport.receive_exactly(
            payload_size + CRC_SIZE
        )

        packet = header + remaining

        # 5. 驗證封包與 CRC
        frame = parse_frame_packet(packet)

        print("[Server] Packet 解析成功")
        print("[Server] CRC OK")

        # 6. 確認 GRAY8 資料長度
        expected_payload_size = (
            frame["width"] * frame["height"]
        )

        if frame["payload_size"] != expected_payload_size:
            raise ValueError(
                "GRAY8 Payload 長度與影像尺寸不符："
                f"{frame['payload_size']} != "
                f"{expected_payload_size}"
            )

        # 7. 將 Payload 直接還原成灰階陣列
        gray = np.frombuffer(
            frame["payload"],
            dtype=np.uint8,
        ).reshape(
            frame["height"],
            frame["width"],
        )

        print()
        print("[Image] 灰階影像還原成功")
        print(f"  Shape : {gray.shape}")
        print(f"  Bytes : {gray.nbytes}")

        # 8. 灰階轉成 YOLO 所需的三通道影像
        yolo_image = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2RGB,
        )

        # 9. 執行 YOLO 推論
        print()
        print("[YOLO] 開始推論...")

        start_time = time.perf_counter()

        results = model.predict(
            source=yolo_image,
            device=0,
            verbose=False,
        )

        inference_elapsed = (
            time.perf_counter() - start_time
        ) * 1000

        result = results[0]

        # 10. 顯示偵測結果
        print()
        print("========================================")
        print("          INFERENCE SUCCESS")
        print("========================================")

        print(f"Frame ID      : {frame['frame_id']}")
        print(
            f"Image Size    : "
            f"{frame['width']} x {frame['height']}"
        )
        print(f"Detected Count: {len(result.boxes)}")
        print(f"Inference Time: {inference_elapsed:.2f} ms")

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            class_name = model.names[class_id]

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            print(
                f"  Class: {class_name}, "
                f"Confidence: {confidence:.3f}, "
                f"Box: "
                f"({x1:.1f}, {y1:.1f}, "
                f"{x2:.1f}, {y2:.1f})"
            )

        # 11. 儲存標記結果
        result.save(filename=OUTPUT_PATH)

        print()
        print(f"[YOLO] 結果圖片：{OUTPUT_PATH}")
        print("[Server] 單張影像處理完成")

    finally:
        # 12. 關閉連線
        if client_transport is not None:
            client_transport.close()

        if server_transport is not None:
            server_transport.close()

        print("[Server] 連線已關閉")


if __name__ == "__main__":
    main()
