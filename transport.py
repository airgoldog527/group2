import socket


class TCPTransport:
    """
    TCP 通訊模組。

    可以作為 Client 或 Server 使用。
    """

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None

    # ========================================================
    # Client
    # ========================================================

    def connect(self):
        """作為 Client 連線到 Server。"""

        if self.sock is not None:
            raise RuntimeError("Socket 已經建立")

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        if self.timeout is not None:
            self.sock.settimeout(self.timeout)

        self.sock.connect(
            (self.host, self.port)
        )

    # ========================================================
    # Server
    # ========================================================

    def listen(self, backlog=1):
        """
        作為 Server 等待 Client 連線。

        Returns
        -------
        TCPTransport
            新建立的 Client Transport。
        """

        if self.sock is not None:
            raise RuntimeError("Socket 已經建立")

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        # 允許程式重啟後快速重新使用 Port
        self.sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        self.sock.bind(
            (self.host, self.port)
        )

        self.sock.listen(backlog)

        print(
            f"[TCP Server] 等待連線 "
            f"{self.host}:{self.port}"
        )

        client_socket, address = self.sock.accept()

        print(
            f"[TCP Server] Client 已連線：{address}"
        )

        client = TCPTransport(
            self.host,
            self.port,
            self.timeout
        )

        client.sock = client_socket

        return client

    # ========================================================
    # Send
    # ========================================================

    def send(self, data):
        """傳送完整 bytes。"""

        if self.sock is None:
            raise RuntimeError("尚未建立 Socket")

        if not isinstance(data, bytes):
            raise TypeError("data 必須是 bytes")

        self.sock.sendall(data)

    # ========================================================
    # Receive Exactly
    # ========================================================

    def receive_exactly(self, size):
        """
        接收指定大小的 bytes。

        TCP 是 Stream，
        因此一次 recv() 不保證收到完整資料。
        """

        if self.sock is None:
            raise RuntimeError("尚未建立 Socket")

        data = bytearray()

        while len(data) < size:

            chunk = self.sock.recv(
                size - len(data)
            )

            if not chunk:
                raise ConnectionError(
                    "TCP 連線在資料接收完成前中斷"
                )

            data.extend(chunk)

        return bytes(data)

    # ========================================================
    # Close
    # ========================================================

    def close(self):
        """關閉 Socket。"""

        if self.sock is not None:

            self.sock.close()

            self.sock = None