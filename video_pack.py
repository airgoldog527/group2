import cv2
import os

from protocol import (
    create_frame_packet,
    PIXEL_FORMAT_GRAY8,
    PIXEL_FORMAT_BGR24
)


# =========================
# 基本設定
# =========================

VIDEO_PATH = "video/test.mp4"

# 每秒取幾幀
SAMPLE_FPS = 5

# 封包輸出資料夾
COLOR_FOLDER = "frames/color"
GRAY_FOLDER = "frames/gray"


# =========================
# 建立資料夾
# =========================

os.makedirs(COLOR_FOLDER, exist_ok=True)
os.makedirs(GRAY_FOLDER, exist_ok=True)


# =========================
# 開啟影片
# =========================

print("正在開啟影片...")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("影片開啟失敗！")
    exit()

print("影片開啟成功！")


# =========================
# 取得影片資訊
# =========================

SOURCE_FPS = cap.get(cv2.CAP_PROP_FPS)

TOTAL_FRAMES = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

VIDEO_SECONDS = TOTAL_FRAMES / SOURCE_FPS


print()
print("===== 影片資訊 =====")
print("原始 FPS:", SOURCE_FPS)
print("總幀數:", TOTAL_FRAMES)
print("影片長度:", VIDEO_SECONDS, "秒")
print("抽取 FPS:", SAMPLE_FPS)


# =========================
# 每秒取 SAMPLE_FPS 幀
# =========================

frame_id = 1
sample_index = 0

while True:

    # 計算這次要取得的原始影片 frame 編號
    target_frame = int(
        round(
            sample_index * SOURCE_FPS / SAMPLE_FPS
        )
    )

    # 超過影片範圍就結束
    if target_frame >= TOTAL_FRAMES:
        break

    # 移動到指定 frame
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        target_frame
    )

    ret, frame = cap.read()

    if not ret:
        print("讀取影片失敗！")
        break


    # =========================
    # 取得寬高
    # =========================

    height, width = frame.shape[:2]


    # ==================================================
    # ① 彩色 Frame Packet
    # ==================================================

    color_payload = frame.tobytes()

    color_packet = create_frame_packet(
        frame_id=frame_id,
        width=width,
        height=height,
        payload=color_payload,
        pixel_format=PIXEL_FORMAT_BGR24
    )

    color_filename = (
        f"frame_{frame_id:04d}.bin"
    )

    color_path = os.path.join(
        COLOR_FOLDER,
        color_filename
    )

    with open(color_path, "wb") as f:
        f.write(color_packet)


    # ==================================================
    # ② 灰階 Frame Packet
    # ==================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray_payload = gray.tobytes()

    gray_packet = create_frame_packet(
        frame_id=frame_id,
        width=width,
        height=height,
        payload=gray_payload,
        pixel_format=PIXEL_FORMAT_GRAY8
    )

    gray_filename = (
        f"frame_{frame_id:04d}.bin"
    )

    gray_path = os.path.join(
        GRAY_FOLDER,
        gray_filename
    )

    with open(gray_path, "wb") as f:
        f.write(gray_packet)


    # =========================
    # 顯示進度
    # =========================

    print(
        f"Frame {frame_id}: "
        f"影片第 {target_frame} 幀 → "
        f"彩色 + 灰階封包完成"
    )


    frame_id += 1
    sample_index += 1


# =========================
# 完成
# =========================

cap.release()

print()
print("================================")
print("影片封包完成！")
print("抽取 FPS:", SAMPLE_FPS)
print("總共處理:", frame_id - 1, "幀")
print("每幀產生：彩色 + 灰階")
print("================================")