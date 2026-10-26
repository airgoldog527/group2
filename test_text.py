import socket
import struct
import threading
import time

print("開始執行 test_text.py")

from config import TCP_PORT, HEADER_FORMAT, HEADER_SIZE, CRC_SIZE
from protocol import (
    create_text_packet,
    parse_text_packet,
)

print("protocol 匯入成功")

HOST = "127.0.0.1"
PORT = TCP_PORT
message = "Eason你拉玩了!!  我是chu???? GGGGGGG Ntnt Reyna are you OK!!!"
test_frame_id = 100

def recv_exact(sock, n_bytes):
    """確保完整接收指定位元組數（處理 TCP 拆包/黏包）"""
    data = bytearray()
    while len(data) < n_bytes:
        chunk = sock.recv(n_bytes - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)

def server_thread_func():
    """本地 Server 端：負責接收並解析文字封包"""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)

    client_sock, _ = server_sock.accept()

    try:
        # 1. 讀取 Header
        header_data = recv_exact(client_sock, HEADER_SIZE)
        if not header_data:
            return

        # 2. 解析 Header 取得 Payload 長度
        header_fields = struct.unpack(HEADER_FORMAT, header_data)
        payload_size = header_fields[7]

        # 3. 讀取 Payload 與 CRC
        tail_data = recv_exact(client_sock, payload_size + CRC_SIZE)
        if not tail_data:
            return

        # 4. 解析封包
        full_packet = header_data + tail_data
        result = parse_text_packet(full_packet)

        print("\n--- [Server] 收到並解析封包 ---")
        print(f"Frame ID：{result['frame_id']}")
        print(f"Payload Size：{result['payload_size']} bytes")
        print(f"Message：{result['message']}")

        # 驗證內容
        assert result["frame_id"] == test_frame_id, "Frame ID 不符"
        assert result["message"] == message, "文字內容不符"
        print("文字驗證成功！ (Server 端 Assert 通過)")

    finally:
        client_sock.close()
        server_sock.close()

def main():
    # 1. 啟動本地接收伺服器 (背景執行緒)
    server = threading.Thread(target=server_thread_func, daemon=True)
    server.start()
    time.sleep(0.3)  # 等待伺服器就緒

    # 2. 建立文字封包
    packet = create_text_packet(
        message=message,
        frame_id=test_frame_id
    )
    print(f"封包建立成功：{len(packet)} bytes")

    # 3. Client 端透過 TCP 發送至本地 127.0.0.1:5000
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect((HOST, PORT))
        client.sendall(packet)
        print("TCP 封包已送出至本地 Server")
    finally:
        client.close()

    # 等候 Server 端處理完成
    server.join(timeout=3.0)
    print("\n本地網路傳輸驗證完成！")

if __name__ == "__main__":
    main()