
# ============================================================
# server_display.py
#
# 第四台電腦：影片與 YOLO 分析結果接收及同步顯示程式
#
# ------------------------------------------------------------
# 【整體系統架構】
#
# 第一台電腦：
#   負責讀取原始影片，並傳送兩路資料：
#
#   1. 彩色影片 BGR24
#      傳送至第四台 Port 5000
#
#   2. 灰階影片 GRAY8
#      傳送至第三台 Port 5000
#
# 第三台電腦：
#   負責接收第一台傳來的灰階影片，
#   使用 YOLO 進行物件辨識。
#
#   辨識完成後，只傳送 YOLO 分析文字及座標，
#   不再將灰階影片轉傳至第四台。
#
# 第四台電腦（本程式）：
#   1. Port 5000：接收第一台的彩色影片。
#   2. Port 5001：接收第三台的 YOLO 分析結果。
#   3. 使用 Frame ID 將兩路資料配對。
#   4. 將 YOLO 辨識框繪製在彩色影片上。
#   5. 顯示同步後的影片。
#
# ------------------------------------------------------------
# 【資料同步方式】
#
# 第一台傳來的影片封包具有 Frame ID。
# 第三台傳來的 YOLO 文字封包也具有 Frame ID。
#
# 例如：
#
#   第一台彩色影片：Frame 100
#   第三台 YOLO 結果：Frame 100
#
# 當兩邊 Frame ID 相同時，
# 就能將該影格的 YOLO 辨識結果畫在對應的影片上。
#
# ============================================================


# ============================================================
# 1. 匯入所需模組
# ============================================================

import socket
import threading
import time
import json

import cv2
import numpy as np

# 從共用設定檔 config.py 取得通訊及影像格式設定
from config import (
    TCP_PORT,
    PIXEL_FORMAT_BGR24,
    PIXEL_FORMAT_RGB24,
)

# StreamReceiver 負責解析 TCP 接收到的封包
from protocol import StreamReceiver


# ============================================================
# 2. 第四台電腦的基本設定
# ============================================================

# 0.0.0.0 代表監聽本機所有網路介面
# 讓其他電腦可以透過網路連線到第四台
HOST = "0.0.0.0"

# ------------------------------------------------------------
# 影片接收 Port
#
# 第一台電腦將彩色影片傳送至第四台的 TCP_PORT。
# ------------------------------------------------------------

VIDEO_PORT = TCP_PORT

# ------------------------------------------------------------
# 文字接收 Port
#
# 第三台電腦將 YOLO 分析結果傳送至第四台。
#
# 使用 TCP_PORT + 1，避免與影片接收 Port 衝突。
# ------------------------------------------------------------

TEXT_PORT = TCP_PORT + 1

# 每次從 TCP Socket 接收資料的最大位元組數
RECV_SIZE = 65536

# ------------------------------------------------------------
# Buffer 緩衝區設定
# ------------------------------------------------------------

# 每個 Buffer 最多保留 120 筆尚未處理的資料
MAX_BUFFER_SIZE = 120

# 資料在 Buffer 中最多保留 5 秒
# 超過時間就會被清除，避免舊資料持續佔用記憶體
BUFFER_TTL = 5.0

# ------------------------------------------------------------
# OpenCV 顯示視窗設定
# ------------------------------------------------------------

# 固定視窗名稱
# 整個程式只使用這一個名稱更新畫面
WINDOW_NAME = "Computer 4 - YOLO Detection"

# True：程式啟動時直接全螢幕
# False：程式啟動時使用一般視窗
START_FULLSCREEN = True


# ============================================================
# 3. 建立共用資料及執行緒同步機制
# ============================================================

# Lock 用來保護多執行緒共用的資料。
#
# 因為：
#   receive_video() 會寫入 video_buffer。
#   receive_text() 會寫入 result_buffer。
#   main() 會讀取並配對兩個 Buffer。
#
# 使用 lock 可以避免不同執行緒同時修改資料，
# 造成資料錯誤或配對問題。
lock = threading.Lock()


# ------------------------------------------------------------
# 影片緩衝區
#
# 用來暫存第一台傳來的彩色影片。
#
# 儲存格式：
#
# video_buffer[frame_id] = (
#     frame,
#     arrival_time
# )
#
# frame：
#   OpenCV 可以處理的彩色影像。
#
# arrival_time：
#   影片抵達第四台的時間。
#
# 例如：
#   video_buffer[100] = (Frame100, 接收時間)
# ------------------------------------------------------------

video_buffer = {}


# ------------------------------------------------------------
# YOLO 結果緩衝區
#
# 用來暫存第三台傳來的 YOLO 分析結果。
#
# 儲存格式：
#
# result_buffer[frame_id] = (
#     objects,
#     arrival_time
# )
#
# objects：
#   YOLO 偵測到的物件資訊及座標。
#
# arrival_time：
#   YOLO 結果抵達第四台的時間。
#
# 例如：
#   result_buffer[100] = (物件清單, 接收時間)
# ------------------------------------------------------------

result_buffer = {}


# ------------------------------------------------------------
# 程式執行狀態
#
# True：
#   程式正常執行。
#
# False：
#   通知接收執行緒及主程式結束。
# ------------------------------------------------------------

running = True


# ------------------------------------------------------------
# 最近一次成功顯示的 Frame ID
#
# 用來避免重複處理已經顯示過的影格。
#
# 初始值 -1：
#   代表目前尚未顯示任何影格。
# ------------------------------------------------------------

last_displayed_frame_id = -1


# ============================================================
# 4. Buffer 清理函式
# ============================================================

def cleanup_buffer(buffer):

    # 取得目前時間
    now = time.monotonic()

    # --------------------------------------------------------
    # 第一階段：清除過期資料
    #
    # 如果資料在 Buffer 中停留超過 BUFFER_TTL 秒，
    # 就將它刪除。
    #
    # 例如：
    #   Frame 100 已經等待超過 5 秒，
    #   但另一邊一直沒有相同 Frame ID，
    #   就不再保留 Frame 100。
    # --------------------------------------------------------

    expired_ids = [
        frame_id
        for frame_id, (_, arrival_time) in buffer.items()
        if now - arrival_time > BUFFER_TTL
    ]

    for frame_id in expired_ids:
        del buffer[frame_id]

    # --------------------------------------------------------
    # 第二階段：限制 Buffer 最大容量
    #
    # 如果資料數量超過 MAX_BUFFER_SIZE，
    # 就刪除 Frame ID 最小的舊資料。
    #
    # 避免長時間執行時，
    # Buffer 不斷增加而佔用過多記憶體。
    # --------------------------------------------------------

    while len(buffer) > MAX_BUFFER_SIZE:

        oldest_id = min(buffer.keys())

        del buffer[oldest_id]


# ============================================================
# 5. 接收第一台傳來的彩色影片
# ============================================================

def receive_video():

    global running

    # --------------------------------------------------------
    # 建立 TCP Server Socket
    # --------------------------------------------------------

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    # 允許程式重新使用已經使用過的 Port
    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    # 綁定第四台的影片接收 Port
    server.bind((HOST, VIDEO_PORT))

    # 開始等待第一台連線
    server.listen(5)

    # 設定 accept 的等待時間
    # 避免程式結束時一直卡在 accept()
    server.settimeout(1.0)

    print(f"[影片] 等待第一台連線：Port {VIDEO_PORT}")

    try:

        # 持續等待第一台連線
        while running:

            try:

                # 接受第一台的 TCP 連線
                conn, addr = server.accept()

            except socket.timeout:

                # 等待逾時時重新檢查 running
                continue

            except OSError:

                if not running:
                    break

                continue

            print(f"[影片] 已連線：{addr}")

            # 每條 TCP 連線建立一個封包接收解析器
            receiver = StreamReceiver()

            with conn:

                # 避免 recv() 永遠阻塞
                conn.settimeout(1.0)

                while running:

                    try:

                        # 從第一台接收 TCP 資料
                        data = conn.recv(RECV_SIZE)

                    except socket.timeout:

                        continue

                    except OSError as e:

                        print(f"[影片] 接收錯誤：{e}")
                        break

                    # 收到空資料代表對方已斷線
                    if not data:
                        break

                    try:

                        # 將收到的 TCP 資料交給解析器
                        receiver.feed(data)

                        # 取出完整的封包
                        packets = receiver.extract_packets()

                    except Exception as e:

                        print(f"[影片] 封包解析錯誤：{e}")
                        continue

                    # ------------------------------------------------
                    # 逐一處理解析完成的封包
                    # ------------------------------------------------

                    for packet in packets:

                        # 只處理 FRAME 類型封包
                        if packet["type_name"] != "FRAME":
                            continue

                        # 取得影格編號
                        frame_id = packet["frame_id"]

                        # 取得影像寬度及高度
                        width = packet["width"]
                        height = packet["height"]

                        # 取得影像格式
                        fmt = packet["pixel_format"]

                        # 取得影像原始資料
                        payload = packet["payload"]

                        try:

                            # ----------------------------------------
                            # BGR24 格式
                            #
                            # 第一台正常傳送的彩色影片格式。
                            #
                            # OpenCV 預設使用 BGR 順序，
                            # 因此可以直接轉換成影像陣列。
                            # ----------------------------------------

                            if fmt == PIXEL_FORMAT_BGR24:

                                expected_size = width * height * 3

                                # 檢查封包資料長度是否正確
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

                            # ----------------------------------------
                            # RGB24 格式
                            #
                            # 如果收到的是 RGB24，
                            # 必須轉換成 OpenCV 使用的 BGR 順序。
                            # ----------------------------------------

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

                            # ----------------------------------------
                            # 其他影像格式
                            #
                            # 第四台不需要接收第三台的灰階影片。
                            # 因此不支援的格式會直接忽略。
                            # ----------------------------------------

                            else:

                                print(
                                    f"[影片] 忽略非彩色封包，"
                                    f"Frame {frame_id}，格式 {fmt}"
                                )

                                continue

                            # ----------------------------------------
                            # 將解碼完成的彩色影像存入 Buffer
                            # ----------------------------------------

                            with lock:

                                # 只保留尚未顯示過的影格
                                if frame_id > last_displayed_frame_id:

                                    video_buffer[frame_id] = (
                                        frame,
                                        time.monotonic()
                                    )

                                    # 清除過期或過多的資料
                                    cleanup_buffer(video_buffer)

                        except Exception as e:

                            print(
                                f"[影片] Frame {frame_id} "
                                f"解碼失敗：{e}"
                            )

            # 連線中斷後，重新等待第一台連線
            print("[影片] 傳送端已斷線，等待重新連線")

    finally:

        # 關閉影片接收 Socket
        server.close()


# ============================================================
# 6. 接收第三台傳來的 YOLO 分析文字
# ============================================================

def receive_text():

    global running

    # --------------------------------------------------------
    # 建立 TCP Server Socket
    # --------------------------------------------------------

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    # 綁定第四台的文字接收 Port
    server.bind((HOST, TEXT_PORT))

    # 開始等待第三台連線
    server.listen(5)

    server.settimeout(1.0)

    print(f"[文字] 等待第三台連線：Port {TEXT_PORT}")

    try:

        while running:

            try:

                # 接受第三台的 TCP 連線
                conn, addr = server.accept()

            except socket.timeout:

                continue

            except OSError:

                if not running:
                    break

                continue

            print(f"[文字] 已連線：{addr}")

            # 建立文字封包解析器
            receiver = StreamReceiver()

            with conn:

                conn.settimeout(1.0)

                while running:

                    try:

                        # 接收第三台傳來的資料
                        data = conn.recv(RECV_SIZE)

                    except socket.timeout:

                        continue

                    except OSError as e:

                        print(f"[文字] 接收錯誤：{e}")
                        break

                    # 對方斷線
                    if not data:
                        break

                    try:

                        # 將收到的 TCP 資料送入解析器
                        receiver.feed(data)

                        # 取得完整封包
                        packets = receiver.extract_packets()

                    except Exception as e:

                        print(f"[文字] 封包解析錯誤：{e}")
                        continue

                    # ------------------------------------------------
                    # 逐一處理文字封包
                    # ------------------------------------------------

                    for packet in packets:

                        # 只處理 TEXT 類型封包
                        if packet["type_name"] != "TEXT":
                            continue

                        # 取得與影片對應的 Frame ID
                        frame_id = packet["frame_id"]

                        # 取得 JSON 文字內容
                        message = packet["message"]

                        try:

                            # ----------------------------------------
                            # 將 JSON 字串轉換成 Python 字典
                            # ----------------------------------------

                            result_data = json.loads(message)

                            if not isinstance(result_data, dict):

                                raise ValueError(
                                    "JSON 格式必須是 dict"
                                )

                            # 取出 YOLO 偵測結果
                            objects = result_data.get(
                                "objects",
                                []
                            )

                            if not isinstance(objects, list):

                                raise ValueError(
                                    "objects 必須是 list"
                                )

                            # ----------------------------------------
                            # 將 YOLO 結果存入 Buffer
                            # ----------------------------------------

                            with lock:

                                # 只保留尚未顯示過的結果
                                if frame_id > last_displayed_frame_id:

                                    result_buffer[frame_id] = (
                                        objects,
                                        time.monotonic()
                                    )

                                    # 清除過期或過多的資料
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

            # 斷線後重新等待第三台連線
            print("[文字] 傳送端已斷線，等待重新連線")

    finally:

        # 關閉文字接收 Socket
        server.close()


# ============================================================
# 7. 依照 Frame ID 取得最新同步影像
# ============================================================

def get_latest_matched_frame():

    global last_displayed_frame_id

    # --------------------------------------------------------
    # 同時讀取兩個 Buffer 時必須取得 Lock
    # 避免接收執行緒同時修改資料。
    # --------------------------------------------------------

    with lock:

        # 清除過期的影片及文字資料
        cleanup_buffer(video_buffer)
        cleanup_buffer(result_buffer)

        # --------------------------------------------
        # 找出兩個 Buffer 都存在的 Frame ID
        #
        # 例如：
        #
        # video_buffer：
        #   100、101、102、103
        #
        # result_buffer：
        #   101、102、104
        #
        # common_ids：
        #   101、102
        #
        # 只有兩邊都有的 Frame ID 才能配對。
        # --------------------------------------------

        common_ids = (
            set(video_buffer.keys())
            &
            set(result_buffer.keys())
        )

        # 如果沒有任何共同 Frame ID
        # 就代表目前沒有成功配對的影格
        if not common_ids:

            return None

        # --------------------------------------------
        # 取得最新一組成功配對的 Frame ID
        # --------------------------------------------

        frame_id = max(common_ids)

        # 取得對應的彩色影片
        frame, _ = video_buffer[frame_id]

        # 取得對應的 YOLO 結果
        objects, _ = result_buffer[frame_id]

        # 複製影片，避免直接修改 Buffer 裡的原始影像
        frame = frame.copy()

        # 更新最後一次顯示的 Frame ID
        last_displayed_frame_id = frame_id

        # --------------------------------------------
        # 清除已經成功處理的舊資料
        #
        # 因為 Frame ID 小於或等於目前顯示的影格，
        # 不需要再保留。
        # --------------------------------------------

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

        # 回傳配對完成的資料
        return frame_id, frame, objects


# ============================================================
# 8. 將 YOLO 偵測結果繪製在彩色影片上
# ============================================================

def draw_yolo_boxes(frame, objects):

    # 取得目前影片的寬度及高度
    height, width = frame.shape[:2]

    # 逐一處理 YOLO 偵測到的物件
    for obj in objects:

        # 確保每筆資料都是字典
        if not isinstance(obj, dict):
            continue

        try:

            # ----------------------------------------
            # 取得物件資訊
            # ----------------------------------------

            class_name = str(
                obj.get("class", "unknown")
            )

            confidence = float(
                obj.get("confidence", 0)
            )

            # 取得物件邊界框座標
            x1 = int(obj.get("x1", 0))
            y1 = int(obj.get("y1", 0))
            x2 = int(obj.get("x2", 0))
            y2 = int(obj.get("y2", 0))

            # ----------------------------------------
            # 限制座標範圍
            #
            # 避免座標超出影像範圍，
            # 造成 OpenCV 繪圖錯誤。
            # ----------------------------------------

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))

            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            # 如果座標不合理，就跳過這筆資料
            if x2 <= x1 or y2 <= y1:
                continue

            # ----------------------------------------
            # 繪製 YOLO 綠色矩形框
            # ----------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # 建立顯示文字
            # 例如：person 0.92
            label = f"{class_name} {confidence:.2f}"

            # ----------------------------------------
            # 計算文字的寬度、高度及基線
            # ----------------------------------------

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2
            )

            # 避免文字超出影像上方
            label_y = max(y1, text_height + 8)

            # 計算文字背景的右側位置
            bg_x2 = min(
                width - 1,
                x1 + text_width + 4
            )

            # 計算文字背景的上方位置
            bg_y1 = max(
                0,
                label_y - text_height - 8
            )

            # 計算文字背景的下方位置
            bg_y2 = min(
                height - 1,
                label_y + baseline
            )

            # ----------------------------------------
            # 繪製綠色文字背景
            # ----------------------------------------

            cv2.rectangle(
                frame,
                (x1, bg_y1),
                (bg_x2, bg_y2),
                (0, 255, 0),
                -1
            )

            # ----------------------------------------
            # 在背景上繪製黑色文字
            # ----------------------------------------

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

            # 如果某筆 YOLO 資料格式錯誤，
            # 就跳過該物件，不影響其他物件的繪製。
            continue

    # 回傳已經繪製 YOLO 框的影片
    return frame


# ============================================================
# 9. 主程式：啟動接收執行緒及同步顯示
# ============================================================

def main():

    global running

    # --------------------------------------------------------
    # 第一階段：建立 OpenCV 顯示視窗
    #
    # namedWindow() 只在程式啟動時呼叫一次。
    #
    # 後續使用 imshow() 更新同一個視窗，
    # 不會在每次迴圈重新建立視窗。
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    # 如果設定為啟動時全螢幕
    if START_FULLSCREEN:

        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

    # --------------------------------------------------------
    # 第二階段：建立影片及文字接收執行緒
    #
    # 影片接收與文字接收分別使用不同執行緒，
    # 避免其中一路等待資料時影響另一條資料流。
    # --------------------------------------------------------

    video_thread = threading.Thread(
        target=receive_video,
        daemon=True
    )

    text_thread = threading.Thread(
        target=receive_text,
        daemon=True
    )

    # 啟動影片接收執行緒
    video_thread.start()

    # 啟動文字接收執行緒
    text_thread.start()

    # --------------------------------------------------------
    # 顯示程式啟動資訊
    # --------------------------------------------------------

    print("=" * 60)
    print("[第四台] YOLO 影片與分析結果同步顯示")
    print(f"[影片] Port {VIDEO_PORT}：第一台彩色影片")
    print(f"[文字] Port {TEXT_PORT}：第三台 YOLO 結果")
    print("[同步] 使用 Frame ID 配對")
    print("[視窗] 僅建立一次")
    print("[全螢幕] 按 F 切換")
    print("[結束] 按 Q")
    print("=" * 60)

    # --------------------------------------------------------
    # 第三階段：顯示狀態初始化
    # --------------------------------------------------------

    # 保存最後一次成功配對的畫面
    #
    # 當暫時沒有新配對結果時，
    # 就繼續顯示這張畫面，避免畫面被清空。
    last_canvas = None

    # 目前顯示的 Frame ID
    display_frame_id = -1

    # 上一次輸出同步成功訊息的 Frame ID
    # 避免重複印出相同資訊。
    last_log_frame_id = -1

    try:

        # ----------------------------------------------------
        # 第四階段：持續執行顯示迴圈
        # ----------------------------------------------------

        while running:

            # --------------------------------------------
            # 從兩個 Buffer 取得最新成功配對的影格
            # --------------------------------------------

            matched = get_latest_matched_frame()

            # --------------------------------------------
            # 情況一：成功取得新的配對影格
            # --------------------------------------------

            if matched is not None:

                # 取得 Frame ID、影片及 YOLO 物件資料
                frame_id, frame, objects = matched

                # 將 YOLO 偵測框畫在彩色影片上
                canvas = draw_yolo_boxes(
                    frame,
                    objects
                )

                # 更新目前顯示的 Frame ID
                display_frame_id = frame_id

                # 保存最後一次成功配對的畫面
                last_canvas = canvas.copy()

                # 只在 Frame ID 改變時輸出同步訊息
                if frame_id != last_log_frame_id:

                    print(
                        f"[同步成功] Frame {frame_id} | "
                        f"物件數量：{len(objects)}"
                    )

                    last_log_frame_id = frame_id

            # --------------------------------------------
            # 情況二：目前沒有新的配對影格
            #
            # 如果之前已經成功配對過，
            # 就繼續顯示上一張畫面。
            #
            # 不清除畫面，也不重新建立視窗。
            # --------------------------------------------

            elif last_canvas is not None:

                canvas = last_canvas.copy()

            # --------------------------------------------
            # 情況三：程式剛啟動
            #
            # 還沒有取得任何成功配對的影格，
            # 因此先顯示黑色等待畫面。
            # --------------------------------------------

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

            # --------------------------------------------
            # 第五階段：在畫面左上角顯示 Frame ID
            # --------------------------------------------

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

            # --------------------------------------------
            # 第六階段：更新 OpenCV 顯示視窗
            #
            # 使用固定的 WINDOW_NAME。
            # 每次只更新同一個視窗的內容。
            # --------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                canvas
            )

            # waitKey() 負責處理鍵盤輸入及視窗事件
            key = cv2.waitKey(1) & 0xFF

            # --------------------------------------------
            # 按 Q：結束程式
            # --------------------------------------------

            if key in (ord("q"), ord("Q")):

                break

            # --------------------------------------------
            # 按 F：切換全螢幕與一般視窗
            # --------------------------------------------

            elif key in (ord("f"), ord("F")):

                # 取得目前視窗顯示模式
                current_mode = cv2.getWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN
                )

                # 如果目前是全螢幕，就切換成一般視窗
                if current_mode == cv2.WINDOW_FULLSCREEN:

                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL
                    )

                # 如果目前不是全螢幕，就切換成全螢幕
                else:

                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN
                    )

            # 稍微讓出 CPU 使用時間
            time.sleep(0.001)

    except KeyboardInterrupt:

        # 使用 Ctrl + C 中斷程式時執行
        print("[第四台] 收到中斷訊號")

    finally:

        # ----------------------------------------------------
        # 第七階段：結束程式及釋放資源
        # ----------------------------------------------------

        # 通知影片及文字接收執行緒停止工作
        running = False

        # 關閉所有 OpenCV 視窗
        cv2.destroyAllWindows()

        print("[第四台] 顯示程式結束")


# ============================================================
# 10. 程式進入點
# ============================================================

# 只有直接執行本檔案時，才會啟動 main()。
#
# 如果本檔案被其他 Python 程式 import，
# 就不會自動啟動第四台的接收及顯示流程。

if __name__ == "__main__":

    main()
