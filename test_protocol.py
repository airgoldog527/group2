
# test_protocol.py
# 封包協定單元測試：灰階、RGB、BGR、文字及連續封包

import unittest

from protocol import (
    create_frame_packet,
    create_text_packet,
    parse_packet,
    parse_frame_packet,
    parse_text_packet,
    StreamReceiver,
)

from config import (
    PIXEL_FORMAT_GRAY8,
    PIXEL_FORMAT_RGB24,
    PIXEL_FORMAT_BGR24,
)


class TestProtocol(unittest.TestCase):

    def test_gray_frame(self):
        """測試灰階影像封包。"""
        width = 8
        height = 4
        frame_id = 1
        payload = bytes(range(width * height))

        packet = create_frame_packet(
            frame_id=frame_id,
            width=width,
            height=height,
            payload=payload,
            pixel_format=PIXEL_FORMAT_GRAY8,
        )

        result = parse_frame_packet(packet)

        self.assertEqual(result["frame_id"], frame_id)
        self.assertEqual(result["width"], width)
        self.assertEqual(result["height"], height)
        self.assertEqual(result["payload"], payload)

    def test_rgb_frame(self):
        """測試 RGB 彩色影像封包。"""
        width = 4
        height = 3
        payload = bytes(width * height * 3)

        packet = create_frame_packet(
            frame_id=2,
            width=width,
            height=height,
            payload=payload,
            pixel_format=PIXEL_FORMAT_RGB24,
        )

        result = parse_frame_packet(packet)

        self.assertEqual(result["pixel_format"], PIXEL_FORMAT_RGB24)
        self.assertEqual(result["payload"], payload)

    def test_bgr_frame(self):
        """測試 BGR 彩色影像封包。"""
        width = 4
        height = 3
        payload = bytes(width * height * 3)

        packet = create_frame_packet(
            frame_id=3,
            width=width,
            height=height,
            payload=payload,
            pixel_format=PIXEL_FORMAT_BGR24,
        )

        result = parse_frame_packet(packet)

        self.assertEqual(result["pixel_format"], PIXEL_FORMAT_BGR24)
        self.assertEqual(result["payload"], payload)

    def test_text_packet(self):
        """測試包含繁體中文字的文字封包。"""
        message = "FPGA 影像辨識測試成功"

        packet = create_text_packet(message)
        result = parse_text_packet(packet)

        self.assertEqual(result["message"], message)

    def test_invalid_crc(self):
        """測試封包資料遭到修改時，解析器是否拒絕封包。"""
        packet = bytearray(
            create_frame_packet(
                frame_id=4,
                width=2,
                height=2,
                payload=b"\x01\x02\x03\x04",
                pixel_format=PIXEL_FORMAT_GRAY8,
            )
        )

        # 修改 CRC 前一個位元組，使資料與原本的 CRC 不一致
        packet[-5] ^= 0x01

        with self.assertRaises(Exception):
            parse_packet(bytes(packet))

    def test_multiple_packets(self):
        """測試多個封包連續到達時，能否分別解析。"""
        packet1 = create_frame_packet(
            frame_id=10,
            width=2,
            height=2,
            payload=b"\x01\x02\x03\x04",
            pixel_format=PIXEL_FORMAT_GRAY8,
        )

        packet2 = create_text_packet("第二個封包")

        receiver = StreamReceiver()

        # 模擬兩個封包一次連續到達
        receiver.feed(packet1 + packet2)
        results = list(receiver.extract_packets())

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["payload"], b"\x01\x02\x03\x04")
        self.assertEqual(results[1]["message"], "第二個封包")

    def test_fragmented_packet(self):
        """測試單一封包分成多次收到時，能否正確組合。"""
        payload = b"\x10\x20\x30\x40"

        packet = create_frame_packet(
            frame_id=11,
            width=2,
            height=2,
            payload=payload,
            pixel_format=PIXEL_FORMAT_GRAY8,
        )

        receiver = StreamReceiver()
        split_at = len(packet) // 2

        # 第一次只收到半個封包
        receiver.feed(packet[:split_at])
        first_results = list(receiver.extract_packets())
        self.assertEqual(len(first_results), 0)

        # 第二次收到剩餘資料
        receiver.feed(packet[split_at:])
        second_results = list(receiver.extract_packets())

        self.assertEqual(len(second_results), 1)
        self.assertEqual(second_results[0]["payload"], payload)

if __name__ == "__main__":
    class CustomTestRunner(unittest.TextTestRunner):
        def run(self, test):
            result = super().run(test)
            if result.wasSuccessful():
                self.stream.writeln("==============================\nTransport Test PASSED!\n==============================")
            return result

    unittest.main(verbosity=0, testRunner=CustomTestRunner)