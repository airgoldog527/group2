# ============================================================
# client_text.py
# 負責將 YOLO 分析文字封包傳送至電腦 C
# TCP Port: 5001
# ============================================================

import json

from config import TARGET_IP, TARGET_PORT
from protocol import create_text_packet
from transport import TCPTransport


class YOLOTextSender:
    """
    YOLO 文字封包傳送端。

    電腦 B 使用此類別連線至電腦 C 的 TCP 5001，
    傳送 YOLO 分析後的 TEXT 封包。
    """

    def __init__(
        self,
        target_ip=TARGET_IP,
        target_port=TARGET_PORT,
    ):
        self.target_ip = target_ip
        self.target_port = target_port

        self.client = TCPTransport(
            host=self.target_ip,
            port=self.target_port,
            timeout=5.0,
        )

        self.connected = False

    def start_connection(self):
        """建立與電腦 C 的 TCP 連線。"""

        if self.connected:
            return True

        try:
            print(
                f"[TEXT] 正在連線至 "
                f"{self.target_ip}:{self.target_port}"
            )

            self.client.connect()
            self.connected = True

            print("[TEXT] 文字傳送端連線成功")

            return True

        except Exception as e:
            print(f"[TEXT] 連線失敗：{e}")
            self.connected = False

            self.client.close()

            return False

    def send_yolo_packet(self, packet):
        """
        傳送已建立好的 YOLO TEXT 封包。

        Parameters
        ----------
        packet : bytes
            由 create_text_packet() 建立的完整封包。
        """

        if not self.connected:
            raise ConnectionError(
                "文字傳送端尚未建立連線"
            )

        if not isinstance(packet, bytes):
            raise TypeError(
                "YOLO 文字封包必須是 bytes"
            )

        self.client.send(packet)

    def send_yolo_results(self, frame_id, objects):
        """
        將 YOLO 偵測結果轉成 TEXT 封包並傳送。

        Parameters
        ----------
        frame_id : int
            對應的影像 Frame ID。

        objects : list
            YOLO 偵測結果，例如：

            [
                {
                    "class": "person",
                    "confidence": 0.95,
                    "bbox": [100, 50, 30, 80]
                }
            ]
        """

        if not isinstance(frame_id, int):
            raise TypeError("frame_id 必須是 int")

        if not isinstance(objects, list):
            raise TypeError("objects 必須是 list")

        # 建立與第四台接收程式相容的 JSON 格式
        result_data = {
            "objects": objects
        }

        message = json.dumps(
            result_data,
            ensure_ascii=False
        )

        # 建立完整 TEXT 封包，封包 Header 也會帶有 frame_id
        packet = create_text_packet(
            message=message,
            frame_id=frame_id
        )

        # 傳送封包
        self.send_yolo_packet(packet)

        print(
            f"[TEXT] 已傳送 YOLO 結果 "
            f"Frame ID={frame_id} "
            f"物件數={len(objects)} "
            f"封包大小={len(packet)} bytes"
        )

    def close_connection(self):
        """關閉文字傳送連線。"""

        self.client.close()
        self.connected = False

        print("[TEXT] 文字傳送連線已關閉")