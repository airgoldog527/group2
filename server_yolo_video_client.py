
import time
import cv2
import numpy as np
import torch

from ultralytics import YOLO

from config import (
    TCP_PORT,
    HOST_IP,
    TARGET_IP,
    TARGET_PORT,
)

from protocol import (
    StreamReceiver,
    PIXEL_FORMAT_GRAY8,
    create_frame_packet,
)

from transport import TCPTransport
from client_text import YOLOTextSender


# ============================================================
# 本機測試 / 正式連線設定
# ============================================================

# True  ：本機測試，傳送灰階影片與 YOLO 文字
# False ：正式連線，只傳送 YOLO 文字

LOCAL_TEST_MODE = False


if LOCAL_TEST_MODE:

    # --------------------------------------------------------
    # 本機測試目的地
    # --------------------------------------------------------

    VIDEO_DEST_IP = "127.0.0.1"
    VIDEO_DEST_PORT = 5002

    TEXT_DEST_IP = "127.0.0.1"
    TEXT_DEST_PORT = 5003

    print("[設定] 本機測試模式：傳送影像與文字")

else:

    # --------------------------------------------------------
    # 正式連線目的地
    # --------------------------------------------------------

    VIDEO_DEST_IP = None
    VIDEO_DEST_PORT = None

    TEXT_DEST_IP = TARGET_IP
    TEXT_DEST_PORT = TARGET_PORT

    print("[設定] 正式連線模式：只傳送 YOLO 文字")


# ============================================================
# 基本設定
# ============================================================

RECV_SIZE = 65536

MODEL_PATH = "yolo26n.pt"

CONFIDENCE_THRESHOLD = 0.25


# ============================================================
# 主程式
# ============================================================

def main():

    print("=" * 60)
    print("YOLO Video Server")
    print("=" * 60)

    # --------------------------------------------------------
    # 顯示目前執行模式
    # --------------------------------------------------------

    if LOCAL_TEST_MODE:
        print("[模式] 本機測試模式")
    else:
        print("[模式] 正式連線模式")

    if LOCAL_TEST_MODE:

        print(
            f"[影像目的地] "
            f"{VIDEO_DEST_IP}:{VIDEO_DEST_PORT}"
        )

    else:

        print("[影像目的地] 不傳送灰階影片")

    print(
        f"[文字目的地] "
        f"{TEXT_DEST_IP}:{TEXT_DEST_PORT}"
    )

    print(
        f"[伺服器監聽] "
        f"{HOST_IP}:{TCP_PORT}"
    )

    # --------------------------------------------------------
    # Socket 物件初始化
    # --------------------------------------------------------

    text_sender = None
    video_sender = None
    server_transport = None
    client_transport = None

    try:

        # ----------------------------------------------------
        # 1. 載入 YOLO 模型
        # ----------------------------------------------------

        print("\n[YOLO] 正在載入模型...")

        model = YOLO(MODEL_PATH)

        if torch.cuda.is_available():

            device = 0
            print("[YOLO] 使用 GPU")

        else:

            device = "cpu"
            print("[YOLO] 使用 CPU")

        print("[YOLO] 模型載入完成")

        # ----------------------------------------------------
        # 2. 建立文字傳送連線
        # ----------------------------------------------------

        try:

            text_sender = YOLOTextSender(
                target_ip=TEXT_DEST_IP,
                target_port=TEXT_DEST_PORT,
            )

            if text_sender.start_connection():

                print("[文字] 連線成功")

            else:

                print("[文字] 連線失敗，暫停文字傳送")
                text_sender = None

        except Exception as e:

            print(f"[文字] 連線錯誤：{e}")
            text_sender = None

        # ----------------------------------------------------
        # 3. 建立影像傳送連線
        #    只有本機測試模式才建立
        # ----------------------------------------------------

        if LOCAL_TEST_MODE:

            try:

                video_sender = TCPTransport(
                    VIDEO_DEST_IP,
                    VIDEO_DEST_PORT,
                    timeout=5.0,
                )

                video_sender.connect()

                print("[影像] 本機測試連線成功")

            except Exception as e:

                print(f"[影像] 本機測試連線錯誤：{e}")

                if video_sender is not None:

                    try:
                        video_sender.close()
                    except Exception:
                        pass

                video_sender = None

        else:

            print("[影像] 正式模式不傳送灰階影片")

        # ----------------------------------------------------
        # 4. 建立第一台電腦的影像接收伺服器
        # ----------------------------------------------------

        server_transport = TCPTransport(
            HOST_IP,
            TCP_PORT,
        )

        print("\n[伺服器] 等待第一台電腦傳送影像...")

        print(
            f"[伺服器] 監聽位置："
            f"{HOST_IP}:{TCP_PORT}"
        )

        # listen() 回傳的是已接受連線的 TCPTransport
        client_transport = server_transport.listen()

        print("[伺服器] 第一台電腦已連線")

        # ----------------------------------------------------
        # 5. 建立封包接收器
        # ----------------------------------------------------

        receiver = StreamReceiver()

        frame_count = 0
        start_time = time.time()

        # ----------------------------------------------------
        # 6. 持續接收第一台傳來的影像
        # ----------------------------------------------------

        while True:

            data = client_transport.sock.recv(RECV_SIZE)

            if not data:

                print("[接收] 第一台電腦已中斷連線")
                break

            receiver.feed(data)

            for packet in receiver.extract_packets():

                # --------------------------------------------
                # 只處理 FRAME 封包
                # --------------------------------------------

                if packet["packet_type"] != 0x01:
                    continue

                frame_id = packet["frame_id"]

                width = packet["width"]
                height = packet["height"]

                pixel_format = packet["pixel_format"]

                payload = packet["payload"]

                # --------------------------------------------
                # 目前只處理 GRAY8
                # --------------------------------------------

                if pixel_format != PIXEL_FORMAT_GRAY8:

                    print(
                        f"[影像] Frame {frame_id} "
                        f"不支援的像素格式：{pixel_format}"
                    )

                    continue

                expected_size = width * height

                if len(payload) != expected_size:

                    print(
                        f"[影像] Frame {frame_id} "
                        f"資料長度錯誤"
                    )

                    continue

                # --------------------------------------------
                # 轉換成 OpenCV 灰階影像
                # --------------------------------------------

                gray = np.frombuffer(
                    payload,
                    dtype=np.uint8,
                ).reshape(height, width)

                # --------------------------------------------
                # 7. 傳送灰階影像
                #    只有本機測試模式才傳送
                # --------------------------------------------

                if LOCAL_TEST_MODE and video_sender is not None:

                    try:

                        frame_packet = create_frame_packet(
                            frame_id=frame_id,
                            width=width,
                            height=height,
                            payload=payload,
                            pixel_format=PIXEL_FORMAT_GRAY8,
                        )

                        video_sender.send(frame_packet)

                    except Exception as e:

                        print(
                            f"[影像] Frame {frame_id} "
                            f"傳送失敗：{e}"
                        )

                        try:
                            video_sender.close()
                        except Exception:
                            pass

                        video_sender = None

                # --------------------------------------------
                # 8. YOLO 推論
                # --------------------------------------------

                yolo_image = cv2.cvtColor(
                    gray,
                    cv2.COLOR_GRAY2BGR,
                )

                results = model.predict(
                    source=yolo_image,
                    device=device,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                )

                result = results[0]

                # --------------------------------------------
                # 9. 整理 YOLO 偵測結果
                # --------------------------------------------

                detected_items = []

                if result.boxes is not None:

                    for box in result.boxes:

                        class_id = int(
                            box.cls[0].item()
                        )

                        confidence = float(
                            box.conf[0].item()
                        )

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0].tolist(),
                        )

                        class_name = model.names[class_id]

                        detected_items.append({

                            "class": class_name,

                            "confidence": round(
                                confidence,
                                4,
                            ),

                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,

                        })

                # --------------------------------------------
                # 10. 傳送 YOLO 文字結果
                # --------------------------------------------

                if text_sender is not None:

                    try:

                        text_sender.send_yolo_results(
                            frame_id=frame_id,
                            objects=detected_items,
                        )

                        print(
                            f"[文字] 已傳送 YOLO 結果 "
                            f"Frame ID={frame_id} "
                            f"物件數={len(detected_items)}"
                        )

                    except Exception as e:

                        print(
                            f"[文字] Frame {frame_id} "
                            f"傳送失敗：{e}"
                        )

                        try:
                            text_sender.close_connection()
                        except Exception:
                            pass

                        text_sender = None

                # --------------------------------------------
                # 11. 第三台本機顯示 YOLO 結果
                # --------------------------------------------

                overlay = result.plot()

                frame_count += 1

                elapsed = time.time() - start_time

                fps = (
                    frame_count / elapsed
                    if elapsed > 0
                    else 0
                )

                cv2.putText(
                    overlay,
                    f"Frame ID: {frame_id}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    overlay,
                    f"FPS: {fps:.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(
                    "YOLO Server Display",
                    overlay,
                )

                print(
                    f"[YOLO] Frame {frame_id} "
                    f"偵測到 {len(detected_items)} 個物件"
                )

                # --------------------------------------------
                # 按 Q 結束
                # --------------------------------------------

                if cv2.waitKey(1) & 0xFF == ord("q"):

                    print("[系統] 使用者結束程式")
                    return

    except KeyboardInterrupt:

        print("\n[系統] 使用者中斷程式")

    except Exception as e:

        print(f"\n[錯誤] 伺服器執行錯誤：{e}")

    finally:

        # ----------------------------------------------------
        # 12. 關閉所有連線
        # ----------------------------------------------------

        print("\n[系統] 正在關閉連線...")

        if text_sender is not None:

            try:
                text_sender.close_connection()
            except Exception:
                pass

        if video_sender is not None:

            try:
                video_sender.close()
            except Exception:
                pass

        if client_transport is not None:

            try:
                client_transport.close()
            except Exception:
                pass

        if server_transport is not None:

            try:
                server_transport.close()
            except Exception:
                pass

        cv2.destroyAllWindows()

        print("[系統] 所有連線已關閉")


# ============================================================
# 程式進入點
# ============================================================

if __name__ == "__main__":
    main()
