import threading
import time
import struct

import cv2

from protocol import (
    create_frame_packet,
    parse_frame_packet,
    HEADER_SIZE,
    CRC_SIZE,
)

from transport import TCPTransport

from config import TCP_PORT,HEADER_FORMAT

# ============================================================
# Test Settings
# ============================================================

HOST = "127.0.0.1"

IMAGE_PATH = "image/test.jpg"

FRAME_ID = 1


# ============================================================
# Server
# ============================================================

def server():

    server_transport = TCPTransport(
        HOST,
        TCP_PORT
    )

    client_transport = server_transport.listen()

    # --------------------------------------------------------
    # 先接收 Header
    # --------------------------------------------------------

    header = client_transport.receive_exactly(
        HEADER_SIZE
    )

    # --------------------------------------------------------
    # 從 Header 取得 Payload Size
    # --------------------------------------------------------


    (
        magic,
        version,
        packet_type,
        frame_id,
        width,
        height,
        pixel_format,
        payload_size
    ) = struct.unpack(
        HEADER_FORMAT,
        header
    )

    print()
    print("[Server]")
    print(f"  Frame ID     : {frame_id}")
    print(f"  Width        : {width}")
    print(f"  Height       : {height}")
    print(f"  Payload Size : {payload_size}")

    # --------------------------------------------------------
    # 接收 Payload + CRC
    # --------------------------------------------------------

    remaining_size = payload_size + CRC_SIZE

    remaining = client_transport.receive_exactly(
        remaining_size
    )

    # --------------------------------------------------------
    # 組回完整 Packet
    # --------------------------------------------------------

    packet = header + remaining

    # --------------------------------------------------------
    # 使用 protocol.py 解析
    # --------------------------------------------------------

    frame = parse_frame_packet(packet)

    print("[Server] Packet 解析成功")
    print(f"[Server] CRC OK")

    # --------------------------------------------------------
    # 還原影像
    # --------------------------------------------------------

    import numpy as np

    payload = frame["payload"]

    received_image = np.frombuffer(
        payload,
        dtype=np.uint8
    ).reshape(
        frame["height"],
        frame["width"]
    )

    # --------------------------------------------------------
    # 儲存收到的影像
    # --------------------------------------------------------

    cv2.imwrite(
        "received_frame.jpg",
        received_image
    )

    print("[Server] 影像已儲存：received_frame.jpg")

    # --------------------------------------------------------
    # 關閉
    # --------------------------------------------------------

    client_transport.close()
    server_transport.close()


# ============================================================
# Client
# ============================================================

def client():

    # --------------------------------------------------------
    # 讀取圖片
    # --------------------------------------------------------

    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:
        raise FileNotFoundError(
            f"找不到圖片：{IMAGE_PATH}"
        )

    # --------------------------------------------------------
    # BGR → GRAY8
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    print("[Client]")
    print(f"  Image Size : {width} × {height}")

    # --------------------------------------------------------
    # GRAY8 → bytes
    # --------------------------------------------------------

    payload = gray.tobytes()

    # --------------------------------------------------------
    # 建立 Frame Packet
    # --------------------------------------------------------

    packet = create_frame_packet(
        frame_id=FRAME_ID,
        width=width,
        height=height,
        payload=payload
    )

    print(f"  Payload    : {len(payload)} bytes")
    print(f"  Packet     : {len(packet)} bytes")

    # --------------------------------------------------------
    # TCP
    # --------------------------------------------------------

    transport = TCPTransport(
        HOST,
        TCP_PORT
    )

    print("[Client] 正在連線...")

    transport.connect()

    print("[Client] 連線成功")

    # --------------------------------------------------------
    # 傳送完整 Packet
    # --------------------------------------------------------

    transport.send(packet)

    print("[Client] Packet 傳送完成")

    transport.close()


# ============================================================
# Main
# ============================================================

def main():

    print("========================================")
    print("       Frame Transfer Test")
    print("========================================")

    # 啟動 Server
    server_thread = threading.Thread(
        target=server
    )

    server_thread.start()

    # 等 Server 啟動
    time.sleep(0.2)

    # 啟動 Client
    client()

    # 等待 Server 完成
    server_thread.join()

    print()
    print("========================================")
    print("      Frame Transfer Test PASSED!")
    print("========================================")


if __name__ == "__main__":
    main()