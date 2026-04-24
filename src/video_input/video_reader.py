import cv2
import threading
from queue import Full

class VideoReader(threading.Thread):
    """Đọc video trong một luồng riêng, đẩy các frame vào frame_queue."""
    def __init__(self, video_path, frame_queue):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.queue = frame_queue
        self.running = True

    def run(self):
        frame_index = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break  # Video kết thúc
            frame_index += 1
            try:
                # Nếu queue đầy, hàm put sẽ block tối đa 1s rồi Raise queue.Full
                self.queue.put(frame, timeout=1)
            except Full:
                # Nếu sau 1s queue vẫn đầy, bỏ qua frame để tránh đóng băng
                print(f"[VideoReader] Queue full, dropping frame {frame_index}")
                continue
        self.running = False
        self.cap.release()
        print("[VideoReader] Finished reading video.")
