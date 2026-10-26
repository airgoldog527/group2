
import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# 測試設定
# ============================================================

MODEL_PATH = "yolo26n.pt"
IMAGE_PATH = "image/test.jpg"


def main():
    print("========================================")
    print("       YOLO Gray Image Test")
    print("========================================")

    # 1. 讀取測試圖片
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            f"無法讀取圖片：{IMAGE_PATH}"
        )

    # 2. 將 BGR 彩色圖片轉成 GRAY8
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    # 3. 模擬 TCP 收到的 GRAY8 Payload
    payload = gray.tobytes()

    print(f"Frame Size    : {width} x {height}")
    print(f"Payload Size  : {len(payload)} bytes")

    # 4. 模擬由封包 Payload 還原影像
    received_gray = np.frombuffer(
        payload,
        dtype=np.uint8
    ).reshape(height, width)

    # 5. 灰階轉成 YOLO 所需的三通道影像
    yolo_image = cv2.cvtColor(
        received_gray,
        cv2.COLOR_GRAY2RGB
    )

    print(f"GRAY8 Shape   : {received_gray.shape}")
    print(f"YOLO Shape    : {yolo_image.shape}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # 6. 載入 YOLO 模型並執行推論
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=yolo_image,
        device=0,
        verbose=False
    )

    # 7. 顯示偵測結果
    result = results[0]

    print()
    print("========== Inference Result ==========")
    print(f"Frame Size    : {width} x {height}")
    print(f"Detected Count: {len(result.boxes)}")

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        class_name = model.names[class_id]

        print(
            f"Class: {class_name}, "
            f"Confidence: {confidence:.3f}"
        )

    # 8. 儲存標記結果，方便檢查
    output_path = "yolo_gray_result.jpg"

    result.save(filename=output_path)

    print()
    print(f"Result saved  : {output_path}")
    print("Test completed successfully.")


if __name__ == "__main__":
    main()
