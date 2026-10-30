
import socket
import threading
import time
import json

import cv2
import numpy as np

from config import (
    TCP_PORT,
    PIXEL_FORMAT_BGR24,
    PIXEL_FORMAT_RGB24,
)

from protocol import StreamReceiver


# ============================================================
# 第四台設定
# ============================================================

HOST = "0.0.0.0"

# 第一台彩色影片
VIDEO_PORT = TCP_PORT

# 第三台 YOLO 文字
TEXT_PORT = TCP_PORT + 1

RECV_SIZE = 65536

# Buffer 設定
MAX_BUFFER_SIZE = 120
BUFFER_TTL = 5.0

# 顯示視窗名稱
WINDOW_NAME = "Computer 4 - YOLO Detection"

# 啟動時是否直接全螢幕
START_FULLSCREEN = True


# ============================================================
# 共用資料
# ============================================================

lock = threading.Lock()

# video_buffer[frame_id] = (frame, arrival_time)
video_buffer = {}

# result_buffer[frame_id] = (objects, arrival_time)
result_buffer = {}

running = True

# 最近一次成功顯示的 Frame ID
last_displayed_frame_id = -1


# ============================================================
# 清理緩衝區
# ============================================================

def cleanup_buffer(buffer):

    now = time.monotonic()

    # 清除過期資料
    expired_ids = [
        frame_id
        for frame_id, (_, arrival_time) in buffer.items()
        if now - arrival_time > BUFFER_TTL
    ]

    for frame_id in expired_ids:
        del buffer[frame_id]

    # 限制最大暫存數量
    while len(buffer) > MAX_BUFFER_SIZE:

        oldest_id = min(buffer.keys())

        del buffer[oldest_id]


# ============================================================
# 接收第一台彩色影片
# ============================================================

def receive_video():

    global running

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    server.bind((HOST, VIDEO_PORT))
    server.listen(5)
    server.settimeout(1.0)

    print(f"[影片] 等待第一台連線：Port {VIDEO_PORT}")

    try:

        while running:

            try:
                conn, addr = server.accept()

            except socket.timeout:
                continue

            except OSError:

                if not running:
                    break

                continue

            print(f"[影片] 已連線：{addr}")

            receiver = StreamReceiver()

            with conn:

                conn.settimeout(1.0)

                while running:

                    try:
                        data = conn.recv(RECV_SIZE)

                    except socket.timeout:
                        continue

                    except OSError as e:

                        print(f"[影片] 接收錯誤：{e}")
                        break

                    if not data:
                        break

                    try:

                        receiver.feed(data)
                        packets = receiver.extract_packets()

                    except Exception as e:

                        print(f"[影片] 封包解析錯誤：{e}")
                        continue

                    for packet in packets:

                        if packet["type_name"] != "FRAME":
                            continue

                        frame_id = packet["frame_id"]

                        width = packet["width"]
                        height = packet["height"]

                        fmt = packet["pixel_format"]
                        payload = packet["payload"]

                        try:

                            # ------------------------------
                            # 只接收彩色影像
                            # ------------------------------

                            if fmt == PIXEL_FORMAT_BGR24:

                                expected_size = width * height * 3

                                if len(payload) != expected_size:
                                    print(
                                        f"[影片] Frame {frame_id} "
                                        f"資料長度錯誤"
                                    )
                                    continue

                                frame = np.frombuffer(
                                    payload,
                                    dtype=np.uint8
                                ).reshape(
                                    height,
                                    width,
                                    3
                                ).copy()

                            elif fmt == PIXEL_FORMAT_RGB24:

                                expected_size = width * height * 3

                                if len(payload) != expected_size:
                                    print(
                                        f"[影片] Frame {frame_id} "
                                        f"資料長度錯誤"
                                    )
                                    continue

                                rgb = np.frombuffer(
                                    payload,
                                    dtype=np.uint8
                                ).reshape(
                                    height,
                                    width,
                                    3
                                )

                                frame = cv2.cvtColor(
                                    rgb,
                                    cv2.COLOR_RGB2BGR
                                )

                            else:

                                print(
                                    f"[影片] 忽略非彩色封包，"
                                    f"Frame {frame_id}，格式 {fmt}"
                                )

                                continue

                            # ------------------------------
                            # 存入影片 Buffer
                            # ------------------------------

                            with lock:

                                if frame_id > last_displayed_frame_id:

                                    video_buffer[frame_id] = (
                                        frame,
                                        time.monotonic()
                                    )

                                    cleanup_buffer(video_buffer)

                        except Exception as e:

                            print(
                                f"[影片] Frame {frame_id} "
                                f"解碼失敗：{e}"
                            )

            print("[影片] 傳送端已斷線，等待重新連線")

    finally:

        server.close()


# ============================================================
# 接收第三台 YOLO 分析文字
# ============================================================

def receive_text():

    global running

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    server.bind((HOST, TEXT_PORT))
    server.listen(5)
    server.settimeout(1.0)

    print(f"[文字] 等待第三台連線：Port {TEXT_PORT}")

    try:

        while running:

            try:
                conn, addr = server.accept()

            except socket.timeout:
                continue

            except OSError:

                if not running:
                    break

                continue

            print(f"[文字] 已連線：{addr}")

            receiver = StreamReceiver()

            with conn:

                conn.settimeout(1.0)

                while running:

                    try:
                        data = conn.recv(RECV_SIZE)

                    except socket.timeout:
                        continue

                    except OSError as e:

                        print(f"[文字] 接收錯誤：{e}")
                        break

                    if not data:
                        break

                    try:

                        receiver.feed(data)
                        packets = receiver.extract_packets()

                    except Exception as e:

                        print(f"[文字] 封包解析錯誤：{e}")
                        continue

                    for packet in packets:

                        if packet["type_name"] != "TEXT":
                            continue

                        frame_id = packet["frame_id"]
                        message = packet["message"]

                        try:

                            result_data = json.loads(message)

                            if not isinstance(result_data, dict):
                                raise ValueError(
                                    "JSON 格式必須是 dict"
                                )

                            objects = result_data.get(
                                "objects",
                                []
                            )

                            if not isinstance(objects, list):
                                raise ValueError(
                                    "objects 必須是 list"
                                )

                            # ------------------------------
                            # 存入 YOLO 結果 Buffer
                            # ------------------------------

                            with lock:

                                if frame_id > last_displayed_frame_id:

                                    result_buffer[frame_id] = (
                                        objects,
                                        time.monotonic()
                                    )

                                    cleanup_buffer(result_buffer)

                            print(
                                f"[文字] Frame {frame_id} "
                                f"收到 {len(objects)} 個物件"
                            )

                        except (
                            json.JSONDecodeError,
                            ValueError,
                            AttributeError,
                            TypeError
                        ) as e:

                            print(
                                f"[文字] Frame {frame_id} "
                                f"JSON 解析失敗：{e}"
                            )

            print("[文字] 傳送端已斷線，等待重新連線")

    finally:

        server.close()


# ============================================================
# 依照 Frame ID 取得最新同步影像
# ============================================================

def get_latest_matched_frame():

    global last_displayed_frame_id

    with lock:

        cleanup_buffer(video_buffer)
        cleanup_buffer(result_buffer)

        # 找出兩邊都有的 Frame ID
        common_ids = (
            set(video_buffer.keys())
            &
            set(result_buffer.keys())
        )

        if not common_ids:
            return None

        # 取最新一組配對成功的影格
        frame_id = max(common_ids)

        frame, _ = video_buffer[frame_id]
        objects, _ = result_buffer[frame_id]

        frame = frame.copy()

        last_displayed_frame_id = frame_id

        # 清除已處理的舊資料
        old_video_ids = [
            fid for fid in video_buffer
            if fid <= frame_id
        ]

        old_result_ids = [
            fid for fid in result_buffer
            if fid <= frame_id
        ]

        for fid in old_video_ids:
            del video_buffer[fid]

        for fid in old_result_ids:
            del result_buffer[fid]

        return frame_id, frame, objects


# ============================================================
# 繪製 YOLO 框
# ============================================================

def draw_yolo_boxes(frame, objects):

    height, width = frame.shape[:2]

    for obj in objects:

        if not isinstance(obj, dict):
            continue

        try:

            class_name = str(
                obj.get("class", "unknown")
            )

            confidence = float(
                obj.get("confidence", 0)
            )

            x1 = int(obj.get("x1", 0))
            y1 = int(obj.get("y1", 0))
            x2 = int(obj.get("x2", 0))
            y2 = int(obj.get("y2", 0))

            # 限制座標範圍
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))

            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"{class_name} {confidence:.2f}"

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2
            )

            label_y = max(y1, text_height + 8)

            bg_x2 = min(
                width - 1,
                x1 + text_width + 4
            )

            bg_y1 = max(
                0,
                label_y - text_height - 8
            )

            bg_y2 = min(
                height - 1,
                label_y + baseline
            )

            cv2.rectangle(
                frame,
                (x1, bg_y1),
                (bg_x2, bg_y2),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1 + 2, label_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        except (
            TypeError,
            ValueError,
            OverflowError
        ):

            continue

    return frame


# ============================================================
# 主程式
# ============================================================

def main():

    global running

    # ------------------------------
    # 只建立一次 OpenCV 視窗
    # ------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    if START_FULLSCREEN:

        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

    # ------------------------------
    # 啟動接收執行緒
    # ------------------------------

    video_thread = threading.Thread(
        target=receive_video,
        daemon=True
    )

    text_thread = threading.Thread(
        target=receive_text,
        daemon=True
    )

    video_thread.start()
    text_thread.start()

    print("=" * 60)
    print("[第四台] YOLO 影片與分析結果同步顯示")
    print(f"[影片] Port {VIDEO_PORT}：第一台彩色影片")
    print(f"[文字] Port {TEXT_PORT}：第三台 YOLO 結果")
    print("[同步] 使用 Frame ID 配對")
    print("[視窗] 僅建立一次")
    print("[全螢幕] 按 F 切換")
    print("[結束] 按 Q")
    print("=" * 60)

    # 上一次成功配對的畫面
    last_canvas = None

    # 顯示中的 Frame ID
    display_frame_id = -1

    # 避免重複印出相同 Frame ID
    last_log_frame_id = -1

    try:

        while running:

            matched = get_latest_matched_frame()

            # ------------------------------
            # 有新影格配對成功
            # ------------------------------

            if matched is not None:

                frame_id, frame, objects = matched

                canvas = draw_yolo_boxes(
                    frame,
                    objects
                )

                display_frame_id = frame_id

                # 保存最後一張配對成功的畫面
                last_canvas = canvas.copy()

                if frame_id != last_log_frame_id:

                    print(
                        f"[同步成功] Frame {frame_id} | "
                        f"物件數量：{len(objects)}"
                    )

                    last_log_frame_id = frame_id

            # ------------------------------
            # 沒有新影格時保持上一張畫面
            # ------------------------------

            elif last_canvas is not None:

                canvas = last_canvas.copy()

            # ------------------------------
            # 尚未取得第一張配對影像
            # ------------------------------

            else:

                canvas = np.zeros(
                    (800, 1280, 3),
                    dtype=np.uint8
                )

                cv2.putText(
                    canvas,
                    "Waiting for synchronized video...",
                    (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            # ------------------------------
            # 顯示 Frame ID
            # ------------------------------

            cv2.putText(
                canvas,
                f"Frame ID: {display_frame_id}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            # ------------------------------
            # 更新同一個視窗
            # ------------------------------

            cv2.imshow(
                WINDOW_NAME,
                canvas
            )

            key = cv2.waitKey(1) & 0xFF

            # Q：結束
            if key in (ord("q"), ord("Q")):
                break

            # F：切換全螢幕
            elif key in (ord("f"), ord("F")):

                current_mode = cv2.getWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN
                )

                if current_mode == cv2.WINDOW_FULLSCREEN:

                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL
                    )

                else:

                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN
                    )

            time.sleep(0.001)

    except KeyboardInterrupt:

        print("[第四台] 收到中斷訊號")

    finally:

        running = False

        cv2.destroyAllWindows()

        print("[第四台] 顯示程式結束")


# ============================================================
# 程式進入點
# ============================================================

if __name__ == "__main__":

    main()
