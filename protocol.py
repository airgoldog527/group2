"""
Unified packet protocol for the FPGA + YOLO project.

Supported use cases:
1. GRAY8 single frame
2. BGR24 single frame (OpenCV's usual color layout)
3. RGB24 single frame
4. Continuous video: send one FRAME packet per frame, incrementing frame_id.
5. UTF-8 text packet

Wire format:
    Header (19 bytes) + Payload + CRC32 (4 bytes)

CRC32 covers the payload only, matching the existing project implementations.
"""

import struct
import zlib

# Prefer values from the project's config.py, but retain defaults so this file
# can still be imported while config.py is being updated.
try:
    import config as _config
except ImportError:
    _config = None


def _config_value(name, default):
    return getattr(_config, name, default) if _config is not None else default


MAGIC = _config_value("MAGIC", 0x4652414D)  # Existing project value; preserve wire compatibility.
VERSION = _config_value("VERSION", 1)

TYPE_FRAME = _config_value("TYPE_FRAME", 0x01)
TYPE_TEXT = _config_value("TYPE_TEXT", 0x02)

PIXEL_FORMAT_GRAY8 = _config_value("PIXEL_FORMAT_GRAY8", 0x01)
PIXEL_FORMAT_BGR24 = _config_value("PIXEL_FORMAT_BGR24", 0x02)
PIXEL_FORMAT_RGB24 = _config_value("PIXEL_FORMAT_RGB24", 0x03)

HEADER_FORMAT = _config_value("HEADER_FORMAT", "<IBBIHHBI")
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
CRC_SIZE = _config_value("CRC_SIZE", 4)

MAX_FRAME_ID = _config_value("MAX_FRAME_ID", 0xFFFFFFFF)
MAX_WIDTH = _config_value("MAX_WIDTH", 0xFFFF)
MAX_HEIGHT = _config_value("MAX_HEIGHT", 0xFFFF)
MAX_PAYLOAD_SIZE = _config_value("MAX_PAYLOAD_SIZE", 0xFFFFFFFF)

_PIXEL_BYTES = {
    PIXEL_FORMAT_GRAY8: 1,
    PIXEL_FORMAT_BGR24: 3,
    PIXEL_FORMAT_RGB24: 3,
}


def _check_uint(name, value, maximum):
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} 必須是整數")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} 超出範圍：{value}（允許 0 到 {maximum}）")


def _as_payload_bytes(payload):
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload 必須是 bytes、bytearray 或 memoryview")
    payload = bytes(payload)
    if len(payload) > MAX_PAYLOAD_SIZE:
        raise ValueError(f"Payload 太大：{len(payload)} bytes")
    return payload


def _build_packet(packet_type, frame_id, width, height, pixel_format, payload):
    payload = _as_payload_bytes(payload)
    _check_uint("frame_id", frame_id, MAX_FRAME_ID)
    _check_uint("width", width, MAX_WIDTH)
    _check_uint("height", height, MAX_HEIGHT)
    _check_uint("pixel_format", pixel_format, 0xFF)

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        VERSION,
        packet_type,
        frame_id,
        width,
        height,
        pixel_format,
        len(payload),
    )
    crc_bytes = struct.pack("<I", zlib.crc32(payload) & 0xFFFFFFFF)
    return header + payload + crc_bytes


# ---------------------------------------------------------------------------
# 1–3. Single-frame packet: GRAY8 / BGR24 / RGB24
# ---------------------------------------------------------------------------

def create_frame_packet(
    frame_id,
    width,
    height,
    payload,
    pixel_format=PIXEL_FORMAT_GRAY8,
):
    """建立一個完整影像封包；payload 必須已是連續排列的像素 bytes。"""
    if pixel_format not in _PIXEL_BYTES:
        raise ValueError(f"不支援的 Pixel Format：{pixel_format}")

    _check_uint("width", width, MAX_WIDTH)
    _check_uint("height", height, MAX_HEIGHT)
    payload = _as_payload_bytes(payload)

    expected_payload_size = width * height * _PIXEL_BYTES[pixel_format]
    if len(payload) != expected_payload_size:
        raise ValueError(
            f"影像 Payload 長度不符：收到 {len(payload)} bytes，"
            f"依 {width}x{height}、pixel_format={pixel_format} "
            f"應為 {expected_payload_size} bytes"
        )

    return _build_packet(
        TYPE_FRAME, frame_id, width, height, pixel_format, payload
    )


def create_gray_frame_packet(frame_id, width, height, payload):
    """建立 GRAY8 單張影像封包（每像素 1 byte）。"""
    return create_frame_packet(
        frame_id, width, height, payload, PIXEL_FORMAT_GRAY8
    )


def create_bgr_frame_packet(frame_id, width, height, payload):
    """建立 BGR24 單張影像封包（每像素 3 bytes，OpenCV 常見排列）。"""
    return create_frame_packet(
        frame_id, width, height, payload, PIXEL_FORMAT_BGR24
    )


def create_rgb_frame_packet(frame_id, width, height, payload):
    """建立 RGB24 單張影像封包（每像素 3 bytes）。"""
    return create_frame_packet(
        frame_id, width, height, payload, PIXEL_FORMAT_RGB24
    )


# ---------------------------------------------------------------------------
# 5. UTF-8 text packet
# ---------------------------------------------------------------------------

def create_text_packet(message, frame_id=0):
    """建立 UTF-8 文字封包；文字的 width/height/pixel_format 欄位皆為 0。"""
    if not isinstance(message, str):
        raise TypeError("message 必須是 str")
    payload = message.encode("utf-8")
    return _build_packet(TYPE_TEXT, frame_id, 0, 0, 0, payload)


# ---------------------------------------------------------------------------
# Unified unpack/parse entry point
# ---------------------------------------------------------------------------

def parse_packet(packet):
    """
    解析一個「已收齊」的完整封包，自動判斷 FRAME 或 TEXT。
    回傳 dict；影像封包包含 width/height/pixel_format，文字封包包含 message。
    """
    if not isinstance(packet, (bytes, bytearray, memoryview)):
        raise TypeError("packet 必須是 bytes、bytearray 或 memoryview")
    packet = bytes(packet)

    if len(packet) < HEADER_SIZE + CRC_SIZE:
        raise ValueError("Packet 太短")

    (
        magic,
        version,
        packet_type,
        frame_id,
        width,
        height,
        pixel_format,
        payload_size,
    ) = struct.unpack(HEADER_FORMAT, packet[:HEADER_SIZE])

    if magic != MAGIC:
        raise ValueError(f"MAGIC 錯誤：0x{magic:X}，預期 0x{MAGIC:X}")
    if version != VERSION:
        raise ValueError(f"VERSION 不支援：{version}")
    if packet_type not in (TYPE_FRAME, TYPE_TEXT):
        raise ValueError(f"未知的封包類型：{packet_type}")
    if payload_size > MAX_PAYLOAD_SIZE:
        raise ValueError(f"Payload 太大：{payload_size} bytes")

    expected_size = HEADER_SIZE + payload_size + CRC_SIZE
    if len(packet) != expected_size:
        raise ValueError(
            f"Packet 長度錯誤：實際 {len(packet)} bytes，預期 {expected_size} bytes"
        )

    payload_start = HEADER_SIZE
    payload_end = payload_start + payload_size
    payload = packet[payload_start:payload_end]

    received_crc = struct.unpack("<I", packet[payload_end:expected_size])[0]
    calculated_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if received_crc != calculated_crc:
        raise ValueError("CRC32 檢查失敗")

    result = {
        "version": version,
        "packet_type": packet_type,
        "frame_id": frame_id,
        "payload_size": payload_size,
        "payload": payload,
    }

    if packet_type == TYPE_FRAME:
        if pixel_format not in _PIXEL_BYTES:
            raise ValueError(f"不支援的 Pixel Format：{pixel_format}")
        if width <= 0 or height <= 0:
            raise ValueError("影像 width 和 height 必須大於 0")

        expected_payload_size = width * height * _PIXEL_BYTES[pixel_format]
        if payload_size != expected_payload_size:
            raise ValueError(
                f"影像 Payload 長度不符：收到 {payload_size} bytes，"
                f"預期 {expected_payload_size} bytes"
            )

        result.update({
            "type_name": "FRAME",
            "width": width,
            "height": height,
            "pixel_format": pixel_format,
        })
    else:
        if width != 0 or height != 0 or pixel_format != 0:
            raise ValueError("文字封包的 width、height、pixel_format 必須為 0")
        try:
            result["message"] = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("文字封包無法以 UTF-8 解碼") from exc
        result["type_name"] = "TEXT"

    return result


def parse_frame_packet(packet):
    """解析影像封包；保留舊版呼叫介面。"""
    result = parse_packet(packet)
    if result["packet_type"] != TYPE_FRAME:
        raise ValueError("Packet TYPE 不是 FRAME")
    return result


def parse_text_packet(packet):
    """解析文字封包；保留舊版呼叫介面。"""
    result = parse_packet(packet)
    if result["packet_type"] != TYPE_TEXT:
        raise ValueError("Packet TYPE 不是 TEXT")
    return result


# ---------------------------------------------------------------------------
# 4. Continuous video over TCP
# ---------------------------------------------------------------------------
# 連續影片不是另一種封包 TYPE：每一影格仍然使用 FRAME packet。
# 傳送端逐幀呼叫 create_frame_packet()，並遞增 frame_id。
# 接收端的 TCP recv() 可能只收到半個封包，也可能一次收到多個封包，
# 因此使用 StreamReceiver 暫存 bytes，等完整封包到齊再解析。

class StreamReceiver:
    """
    TCP 位元組流切包器，支援黏包、分包與連續影格。
    每個連線應建立自己的 StreamReceiver。
    呼叫 feed(data)，再迭代 extract_packets() 取得已解析封包 dict。
    """

    def __init__(self, max_payload_size=None):
        self._buffer = bytearray()
        self.max_payload_size = (
            MAX_PAYLOAD_SIZE if max_payload_size is None else max_payload_size
        )
        if not isinstance(self.max_payload_size, int) or self.max_payload_size < 0:
            raise ValueError("max_payload_size 必須是非負整數")
        # Preserve the existing wire encoding of MAGIC.
        self._magic_bytes = struct.pack("<I", MAGIC)

    def feed(self, data):
        """將 socket.recv() 收到的 bytes 加入緩衝區。"""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data 必須是 bytes、bytearray 或 memoryview")
        if data:
            self._buffer.extend(data)

    def extract_packets(self):
        """
        逐一 yield 已收齊且驗證通過的封包 dict。
        無效資料會丟棄並嘗試重新同步；不完整封包會留在 buffer 等下次 feed。
        """
        while True:
            if len(self._buffer) < HEADER_SIZE:
                return

            magic_index = self._buffer.find(self._magic_bytes)
            if magic_index < 0:
                # 保留可能是跨 recv 邊界的 MAGIC 前綴。
                keep = len(self._magic_bytes) - 1
                if len(self._buffer) > keep:
                    del self._buffer[:-keep]
                return
            if magic_index > 0:
                del self._buffer[:magic_index]
                if len(self._buffer) < HEADER_SIZE:
                    return

            try:
                header = struct.unpack(HEADER_FORMAT, self._buffer[:HEADER_SIZE])
            except struct.error:
                del self._buffer[0]
                continue

            magic, version, packet_type, _, width, height, pixel_format, payload_size = header

            # 不可信或損毀的 Header：往後移一 byte，再找 MAGIC。
            if (
                magic != MAGIC
                or version != VERSION
                or packet_type not in (TYPE_FRAME, TYPE_TEXT)
                or payload_size > self.max_payload_size
            ):
                del self._buffer[0]
                continue

            total_size = HEADER_SIZE + payload_size + CRC_SIZE
            if len(self._buffer) < total_size:
                return

            raw_packet = bytes(self._buffer[:total_size])
            del self._buffer[:total_size]

            try:
                yield parse_packet(raw_packet)
            except (ValueError, TypeError, struct.error):
                # 此封包無效；已切除，繼續尋找下一個封包。
                continue

    @property
    def buffered_bytes(self):
        """目前仍未組成完整封包的緩衝資料量。"""
        return len(self._buffer)
