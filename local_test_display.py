
import socket
import threading
import time
import json

import cv2
import numpy as np

from protocol import (
    StreamReceiver,
    PIXEL_FORMAT_GRAY8,
    PIXEL_FORMAT_BGR24,
    PIXEL_FORMAT_RGB24,
)


# ============================================================
# 1. 本地測試設定
# 不修改 config.py
# ============================================================

HOST = "127.0.0.1"

# 本地模擬第四台接收 Port
VIDEO_PORT = 5002
TEXT_PORT = 5003

RECV_SIZE = 65536

# 緩衝區最多保留幾個尚未配對的 Frame
MAX_BUFFER_SIZE = 100

WINDOW_NAME = "Local Test - Packet Receiver"

DISPLAY_INTERVAL = 0.03

# 是否繼續執行
running = True


# ============================================================
# 2. 共用資料
# ============================================================

data_lock = threading.Lock()

# 依照 frame_id 暫存影像
frame_buffer = {}

# 依照 frame_id 暫存 YOLO 結果
text_buffer = {}

# 統計資訊
stats = {
    "video_packets": 0,
    "text_packets": 0,
    "matched_frames": 0,
    "parse_errors": 0,
    "video_connections": 0,
    "text_connections": 0,
    "last_video_id": None,
    "last_text_id": None,
}


# ============================================================
# 3. 建立 TCP 監聽 Socket
# ============================================================

def create_listener(port):

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    server.bind((HOST, port))

    server.listen(5)

    # 定期檢查 running，避免 accept 永遠阻塞
    server.settimeout(1.0)

    print(f"[監聽] TCP {HOST}:{port}")

    return server


# ============================================================
# 4. 限制緩衝區大小
# ============================================================

def limit_buffer(buffer):

    while len(buffer) > MAX_BUFFER_SIZE:

        oldest_id = next(iter(buffer))

        del buffer[oldest_id]

        print(
            f"[緩衝區] 移除未配對 Frame ID={oldest_id}"
        )


# ============================================================
# 5. 影像封包接收
# ============================================================

def receive_video():

    server = None

    try:

        server = create_listener(VIDEO_PORT)

        while running:

            try:

                client_socket, address = server.accept()

            except socket.timeout:
                continue

            except OSError:
                break

            print(
                f"[影像] 第三台已連線：{address}"
            )

            with data_lock:
                stats["video_connections"] += 1

            client_socket.settimeout(1.0)

            # 每一條 TCP 連線都建立自己的 StreamReceiver
            receiver = StreamReceiver()

            try:

                while running:

                    try:

                        data = client_socket.recv(RECV_SIZE)

                    except socket.timeout:
                        continue

                    if not data:

                        print("[影像] TCP 連線中斷")

                        break

                    # ------------------------------------
                    # 正確的 StreamReceiver 使用方式
                    # ------------------------------------

                    try:

                        receiver.feed(data)

                        packets = receiver.extract_packets()

                    except Exception as e:

                        print(
                            f"[影像] 解封包錯誤：{e}"
                        )

                        with data_lock:
                            stats["parse_errors"] += 1

                        continue

                    # ------------------------------------
                    # 處理完整封包
                    # ------------------------------------

                    for packet in packets:

                        if packet.get("type_name") != "FRAME":
                            continue

                        frame_id = packet["frame_id"]

                        width = packet["width"]

                        height = packet["height"]

                        pixel_format = packet["pixel_format"]

                        payload = packet["payload"]

                        # --------------------------------
                        # GRAY8
                        # --------------------------------

                        if pixel_format == PIXEL_FORMAT_GRAY8:

                            expected_size = width * height

                            if len(payload) != expected_size:

                                print(
                                    f"[影像] Frame {frame_id} "
                                    f"GRAY8 長度錯誤"
                                )

                                with data_lock:
                                    stats["parse_errors"] += 1

                                continue

                            image = np.frombuffer(
                                payload,
                                dtype=np.uint8
                            ).reshape(
                                (height, width)
                            )

                            # 灰階轉 BGR，方便繪製 YOLO 方框
                            image = cv2.cvtColor(
                                image,
                                cv2.COLOR_GRAY2BGR
                            )

                        # --------------------------------
                        # BGR24
                        # --------------------------------

                        elif pixel_format == PIXEL_FORMAT_BGR24:

                            expected_size = width * height * 3

                            if len(payload) != expected_size:

                                print(
                                    f"[影像] Frame {frame_id} "
                                    f"BGR24 長度錯誤"
                                )

                                with data_lock:
                                    stats["parse_errors"] += 1

                                continue

                            image = np.frombuffer(
                                payload,
                                dtype=np.uint8
                            ).reshape(
                                (height, width, 3)
                            ).copy()

                        # --------------------------------
                        # RGB24
                        # --------------------------------

                        elif pixel_format == PIXEL_FORMAT_RGB24:

                            expected_size = width * height * 3

                            if len(payload) != expected_size:

                                print(
                                    f"[影像] Frame {frame_id} "
                                    f"RGB24 長度錯誤"
                                )

                                with data_lock:
                                    stats["parse_errors"] += 1

                                continue

                            image = np.frombuffer(
                                payload,
                                dtype=np.uint8
                            ).reshape(
                                (height, width, 3)
                            )

                            image = cv2.cvtColor(
                                image,
                                cv2.COLOR_RGB2BGR
                            )

                        else:

                            print(
                                f"[影像] 不支援的 Pixel Format："
                                f"{pixel_format}"
                            )

                            with data_lock:
                                stats["parse_errors"] += 1

                            continue

                        # --------------------------------
                        # 儲存影像
                        # --------------------------------

                        with data_lock:

                            frame_buffer[frame_id] = image

                            limit_buffer(frame_buffer)

                            stats["video_packets"] += 1

                            stats["last_video_id"] = frame_id

                        print(
                            f"[影像] 收到 FRAME "
                            f"ID={frame_id} "
                            f"尺寸={width}x{height} "
                            f"Payload={len(payload)} bytes"
                        )

            except Exception as e:

                print(
                    f"[影像] 接收執行緒錯誤：{e}"
                )

                with data_lock:
                    stats["parse_errors"] += 1

            finally:

                client_socket.close()

                print("[影像] 連線已關閉")

    except Exception as e:

        print(
            f"[影像] 伺服器錯誤：{e}"
        )

    finally:

        if server is not None:
            server.close()

        print("[影像] 監聽執行緒結束")


# ============================================================
# 6. YOLO 文字封包接收
# ============================================================

def receive_text():

    server = None

    try:

        server = create_listener(TEXT_PORT)

        while running:

            try:

                client_socket, address = server.accept()

            except socket.timeout:
                continue

            except OSError:
                break

            print(
                f"[文字] 第三台已連線：{address}"
            )

            with data_lock:
                stats["text_connections"] += 1

            client_socket.settimeout(1.0)

            # 文字連線使用獨立 StreamReceiver
            receiver = StreamReceiver()

            try:

                while running:

                    try:

                        data = client_socket.recv(RECV_SIZE)

                    except socket.timeout:
                        continue

                    if not data:

                        print("[文字] TCP 連線中斷")

                        break

                    # ------------------------------------
                    # 正確的 StreamReceiver 使用方式
                    # ------------------------------------

                    try:

                        receiver.feed(data)

                        packets = receiver.extract_packets()

                    except Exception as e:

                        print(
                            f"[文字] 解封包錯誤：{e}"
                        )

                        with data_lock:
                            stats["parse_errors"] += 1

                        continue

                    # ------------------------------------
                    # 處理完整封包
                    # ------------------------------------

                    for packet in packets:

                        if packet.get("type_name") != "TEXT":
                            continue

                        frame_id = packet["frame_id"]

                        message = packet["message"]

                        # --------------------------------
                        # 解析 JSON
                        # --------------------------------

                        try:

                            result_data = json.loads(message)

                            if not isinstance(result_data, dict):

                                raise ValueError(
                                    "JSON 根節點必須是 dict"
                                )

                            objects = result_data.get(
                                "objects",
                                []
                            )

                            if not isinstance(objects, list):

                                raise ValueError(
                                    "objects 必須是 list"
                                )

                        except Exception as e:

                            print(
                                f"[文字] Frame ID={frame_id} "
                                f"JSON 解析失敗：{e}"
                            )

                            with data_lock:
                                stats["parse_errors"] += 1

                            continue

                        # --------------------------------
                        # 儲存 YOLO 結果
                        # --------------------------------

                        with data_lock:

                            text_buffer[frame_id] = objects

                            limit_buffer(text_buffer)

                            stats["text_packets"] += 1

                            stats["last_text_id"] = frame_id

                        print(
                            f"[文字] 收到 TEXT "
                            f"ID={frame_id} "
                            f"物件數={len(objects)}"
                        )

                        for obj in objects:

                            print(
                                "       ",
                                obj
                            )

            except Exception as e:

                print(
                    f"[文字] 接收執行緒錯誤：{e}"
                )

                with data_lock:
                    stats["parse_errors"] += 1

            finally:

                client_socket.close()

                print("[文字] 連線已關閉")

    except Exception as e:

        print(
            f"[文字] 伺服器錯誤：{e}"
        )

    finally:

        if server is not None:
            server.close()

        print("[文字] 監聽執行緒結束")


# ============================================================
# 7. 繪製 YOLO 偵測框
# 支援兩種格式：
#
# 格式 A：
# {
#     "class": "person",
#     "bbox": [x, y, width, height]
# }
#
# 格式 B：
# {
#     "class": "person",
#     "x1": 100,
#     "y1": 50,
#     "x2": 200,
#     "y2": 300
# }
# ============================================================

def draw_yolo_boxes(image, objects):

    height, width = image.shape[:2]

    for obj in objects:

        try:

            class_name = str(
                obj.get("class", "unknown")
            )

            confidence = float(
                obj.get("confidence", 0)
            )

            # --------------------------------
            # 格式 A：bbox [x, y, w, h]
            # --------------------------------

            if "bbox" in obj:

                x, y, w, h = map(
                    int,
                    obj["bbox"]
                )

                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h

            # --------------------------------
            # 格式 B：x1, y1, x2, y2
            # --------------------------------

            elif all(
                key in obj
                for key in ("x1", "y1", "x2", "y2")
            ):

                x1 = int(float(obj["x1"]))
                y1 = int(float(obj["y1"]))
                x2 = int(float(obj["x2"]))
                y2 = int(float(obj["y2"]))

            else:

                print(
                    "[繪圖] 物件缺少 bbox 或 xyxy 座標"
                )

                continue

        except (KeyError, TypeError, ValueError):

            print(
                "[繪圖] 物件資料格式錯誤，略過"
            )

            continue

        # ------------------------------------
        # 限制座標在影像範圍內
        # ------------------------------------

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))

        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        # ------------------------------------
        # 繪製方框
        # ------------------------------------

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return image


# ============================================================
# 8. 顯示統計資訊
# ============================================================

def draw_status(image, frame_id):

    with data_lock:

        current_stats = stats.copy()

    lines = [
        f"Frame ID: {frame_id}",
        f"Video packets: {current_stats['video_packets']}",
        f"Text packets: {current_stats['text_packets']}",
        f"Matched frames: {current_stats['matched_frames']}",
        f"Parse errors: {current_stats['parse_errors']}",
    ]

    y = 25

    for line in lines:

        cv2.putText(
            image,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        y += 25

    return image


# ============================================================
# 9. 主顯示程式
# ============================================================

def main():

    global running

    print("=" * 55)

    print("本地模擬第四台：TCP 封包接收與解碼")

    print("=" * 55)

    print(f"影像監聽：{HOST}:{VIDEO_PORT}")

    print(f"文字監聽：{HOST}:{TEXT_PORT}")

    print("等待第三台傳送影像與 YOLO 結果...")

    print("按 Q 結束程式")

    print("=" * 55)

    # ----------------------------------------
    # 啟動影像接收執行緒
    # ----------------------------------------

    video_thread = threading.Thread(
        target=receive_video,
        daemon=True
    )

    # ----------------------------------------
    # 啟動文字接收執行緒
    # ----------------------------------------

    text_thread = threading.Thread(
        target=receive_text,
        daemon=True
    )

    video_thread.start()

    text_thread.start()

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    last_display_time = 0

    try:

        while running:

            current_time = time.time()

            if (
                current_time - last_display_time
                < DISPLAY_INTERVAL
            ):

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                continue

            last_display_time = current_time

            matched_frame = None

            matched_objects = None

            matched_id = None

            # ------------------------------------
            # 依照 frame_id 配對
            # ------------------------------------

            with data_lock:

                matched_ids = (
                    set(frame_buffer.keys())
                    & set(text_buffer.keys())
                )

                if matched_ids:

                    # 取出目前已配對的最新 Frame ID
                    matched_id = max(matched_ids)

                    matched_frame = frame_buffer.pop(
                        matched_id
                    )

                    matched_objects = text_buffer.pop(
                        matched_id
                    )

                    stats["matched_frames"] += 1

            # ------------------------------------
            # 顯示配對成功的影像
            # ------------------------------------

            if matched_frame is not None:

                display_image = matched_frame.copy()

                # 繪製 YOLO 偵測框
                display_image = draw_yolo_boxes(
                    display_image,
                    matched_objects
                )

                # 顯示統計資訊
                display_image = draw_status(
                    display_image,
                    matched_id
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display_image
                )

                print(
                    f"[配對成功] "
                    f"Frame ID={matched_id} "
                    f"物件數={len(matched_objects)}"
                )

            # ------------------------------------
            # 鍵盤控制
            # ------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:

        print("\n[主程式] 使用者中斷")

    finally:

        running = False

        cv2.destroyAllWindows()

        # ------------------------------------
        # 顯示統計結果
        # ------------------------------------

        print("\n" + "=" * 45)

        print("[本地模擬第四台：統計結果]")

        print("=" * 45)

        with data_lock:

            for key, value in stats.items():

                print(f"{key}: {value}")

        print("=" * 45)

        print("[主程式] 本地模擬第四台結束")


# ============================================================
# 10. 程式進入點
# ============================================================

if __name__ == "__main__":

    main()
