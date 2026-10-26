import threading
import time

from transport import TCPTransport


HOST = "127.0.0.1"
PORT = 5000

TEST_DATA = b"Hello FPGA!"


# ============================================================
# Server
# ============================================================

def server():

    server = TCPTransport(
        HOST,
        PORT
    )

    client = server.listen()

    # 接收資料
    received = client.receive_exactly(
        len(TEST_DATA)
    )

    print(
        f"[Server] 收到：{received!r}"
    )

    # 驗證
    assert received == TEST_DATA

    client.close()
    server.close()


# ============================================================
# Main
# ============================================================

def main():

    # 啟動 Server
    server_thread = threading.Thread(
        target=server
    )

    server_thread.start()

    # 等 Server 啟動
    time.sleep(0.2)

    # Client
    client = TCPTransport(
        HOST,
        PORT
    )

    print("[Client] 正在連線...")

    client.connect()

    print("[Client] 連線成功")

    # 傳送
    client.send(TEST_DATA)

    print("[Client] 資料傳送完成")

    client.close()

    server_thread.join()

    print()
    print("==============================")
    print("Transport Test PASSED!")
    print("==============================")


if __name__ == "__main__":
    main()