import threading
import queue
from src.preprocessing.transform import preprocess_yolo, preprocess_deeplab

class Worker(threading.Thread):
    """
    Thread xử lý frame: lấy từ frame_queue, tiền xử lý,
    đẩy ảnh YOLO và DeepLab vào output_queue.
    """
    def __init__(self, in_q, out_q, worker_id):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.worker_id = worker_id
        self.running = True

    def run(self):
        while self.running:
            try:
                frame = self.in_q.get(timeout=1)  # chờ frame trong 1s
            except queue.Empty:
                # nếu queue rỗng và producer đã dừng, ta cũng dừng
                continue

            # Tiền xử lý cho YOLO và DeepLab
            yolo_img = preprocess_yolo(frame)
            deeplab_img = preprocess_deeplab(frame)

            # Đưa vào output queue (giữ lại frame gốc cho visualization)
            while self.running:
                try:
                    self.out_q.put((frame, yolo_img, deeplab_img), timeout=1)
                    break # Đẩy thành công thì thoát vòng lặp while nhỏ này
                except queue.Full:
                    # Nếu queue đầy, tiếp tục thử lại cho đến khi self.running = False
                    continue
            # Đánh dấu đã xử lý xong frame
            self.in_q.task_done()

        print(f"[Worker {self.worker_id}] Đã dừng")
