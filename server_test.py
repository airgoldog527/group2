import struct
import numpy as np
import cv2

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

HOST = "192.168.1.3"


# ============================================================
# Server
# ============================================================

def main():

    print("========================================")
    print("       Frame Server Test")
    print("========================================")

    # --------------------------------------------------------
    # 建立 TCP Server
    # --------------------------------------------------------

    server_transport = TCPTransport(
        HOST,
        TCP_PORT
    )

    print("[Server] 等待 Client 連線...")

    client_transport = server_transport.listen()

    # --------------------------------------------------------
    # 接收 Header
    # --------------------------------------------------------

    print("[Server] 等待 Header...")

    header = client_transport.receive_exactly(
        HEADER_SIZE
    )

    # --------------------------------------------------------
    # 解析 Header
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
    print("[Server] 收到 Header")
    print(f"  Frame ID     : {frame_id}")
    print(f"  Width        : {width}")
    print(f"  Height       : {height}")
    print(f"  Payload Size : {payload_size}")

    # --------------------------------------------------------
    # 接收 Payload + CRC
    # --------------------------------------------------------

    print("[Server] 等待 Payload + CRC...")

    remaining = client_transport.receive_exactly(
        payload_size + CRC_SIZE
    )

    # --------------------------------------------------------
    # 組回完整 Packet
    # --------------------------------------------------------

    packet = header + remaining

    # --------------------------------------------------------
    # 使用 protocol.py 驗證
    # --------------------------------------------------------

    frame = parse_frame_packet(packet)

    print()
    print("[Server] Packet 解析成功")
    print("[Server] CRC OK")

    # --------------------------------------------------------
    # 顯示結果
    # --------------------------------------------------------

    print()
    print("========================================")
    print("          RECEIVE SUCCESS")
    print("========================================")

    print(f"Frame ID     : {frame['frame_id']}")
    print(f"Image Size   : {frame['width']} × {frame['height']}")
    print(f"Payload Size : {frame['payload_size']} bytes")

    image = np.frombuffer(
    frame["payload"],
    dtype=np.uint8
    ).reshape(
        frame["height"],
        frame["width"]
    )

    cv2.imwrite(
        "received_frame.png",
        image
    )
    # --------------------------------------------------------
    # 關閉
    # --------------------------------------------------------

    client_transport.close()
    server_transport.close()


if __name__ == "__main__":
    main()