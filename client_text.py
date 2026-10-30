from config import TARGET_IP, TARGET_PORT
from transport import TCPTransport


class YOLOTextSender:
    def __init__(self, target_ip=TARGET_IP, target_port=TARGET_PORT):
        self.target_ip = target_ip
        self.target_port = target_port
        self.client = TCPTransport(
            host=self.target_ip,
            port=self.target_port,
            timeout=5.0
        )

    def start_connection(self):
        try:
            print(
                f"[TEXT] 連線至 "
                f"{self.target_ip}:{self.target_port}"
            )
            self.client.connect()
            print("[TEXT] 文字傳送端連線成功")
            return True
        except Exception as e:
            print(f"[TEXT] 連線失敗：{e}")
            return False

    def send_yolo_packet(self, packet):
        self.client.send(packet)

    def close_connection(self):
        self.client.close()
