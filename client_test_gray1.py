
import cv2

from config import TCP_PORT, HOST_IP
from protocol import create_frame_packet, PIXEL_FORMAT_GRAY8
from transport import TCPTransport

# 你的圖片路徑
IMAGE_PATH = "image/boss.jpg"


def main():
    # 讀取圖片
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("圖片讀取失敗！請確認圖片在 image 資料夾內。")
        return

    # 彩色轉灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    payload = gray.tobytes()

    # 建立灰階封包
    packet = create_frame_packet(
        frame_id=1,
        width=width,
        height=height,
        payload=payload,
        pixel_format=PIXEL_FORMAT_GRAY8
    )

    print("圖片已轉成灰階")
    print("圖片尺寸：", width, "x", height)
    print("封包大小：", len(packet), "bytes")

    transport = TCPTransport(HOST_IP, TCP_PORT)

    try:
        print("正在連線到同學的電腦...")
        transport.connect()
        transport.send(packet)
        print("灰階圖片傳送成功！")

    except OSError as e:
        print("傳送失敗：", e)

    finally:
        transport.close()


if __name__ == "__main__":
    main()
