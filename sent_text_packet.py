from config import TCP_PORT
from protocol import create_text_packet
from transport import TCPTransport

# 設定目標 IP 與 Port
# 本機自我測試填 "127.0.0.1"
# 若要傳給朋友，請改為對方的 IP（例如 "192.168.1.3" 或 "192.168.1.2"）
TARGET_IP = "192.168.1.4"
TARGET_PORT = TCP_PORT  # 預設 5000

def main():
    message = "Eason你拉完了!!  我是chu???? GGGGGGG Ntnt Reyna are you OK!!!"
    packet = create_text_packet(message=message, frame_id=1)
    print(f"[*] 封包建立完成（{len(packet)} bytes），準備連線至 {TARGET_IP}:{TARGET_PORT}...")

    client = TCPTransport(host=TARGET_IP, port=TARGET_PORT, timeout=5.0)
    try:
        client.connect()
        print("[*] 連線成功，傳送封包中...")
        client.send(packet)
        print(f"[✓] 成功送出文字封包：{message}")
    except ConnectionRefusedError:
        print(f"[!] 連線失敗：目標 {TARGET_IP}:{TARGET_PORT} 拒絕連線（接收端尚未啟動）。")
    except Exception as e:
        print(f"[!] 傳送時發生錯誤：{e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()