import queue
import time
from configs.config import QUEUE_SIZE, NUM_WORKERS
from src.video_input.video_reader import VideoReader
from src.pipeline.worker import Worker
from src.utils.fps import FPS


class Pipeline:
    def __init__(self, video_path):
        self.frame_q = queue.Queue(maxsize=QUEUE_SIZE)
        self.output_q = queue.Queue(maxsize=QUEUE_SIZE)

        # Producer
        self.reader = VideoReader(video_path, self.frame_q)

        # Consumers
        self.workers = [
            Worker(self.frame_q, self.output_q, i)
            for i in range(NUM_WORKERS)
        ]

        self.fps = FPS()
        self.running = False

    def start(self):
        print("[Pipeline] Khởi động VideoReader và Workers...")
        self.reader.start()

        for w in self.workers:
            w.start()

        self.running = True

    def run(self):
        self.start()
        print("[Pipeline] Đang chạy, nhấn Ctrl+C để dừng...")

        start_time = time.time()

        try:
            while True:
                try:
                    yolo_img, deeplab_img = self.output_q.get(timeout=2)
                except queue.Empty:
                    if not self.reader.running:
                        break
                    continue

                # 👉 Placeholder cho model
                # yolo_model(yolo_img)
                # deeplab_model(deeplab_img)

                self.fps.increment()

                # 🔥 In thông tin mỗi 30 frame
                if self.fps.count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = self.fps.count / elapsed

                    print(f"[INFO] FPS: {fps:.2f}")
                    print(f"[INFO] FrameQ: {self.frame_q.qsize()} | OutputQ: {self.output_q.qsize()}")

                self.output_q.task_done()

        except KeyboardInterrupt:
            print("[Pipeline] Bị dừng bởi người dùng")

        finally:
            self.stop()

            print("\n===== SUMMARY =====")
            print(f"Frames processed: {self.fps.count}")
            print(f"Average FPS: {self.fps.get():.2f}")
            print("===================")

    def stop(self):
        print("[Pipeline] Dừng pipeline...")

        self.reader.running = False

        for w in self.workers:
            w.running = False

        # Đợi thread kết thúc
        self.reader.join()

        for w in self.workers:
            w.join()

        print("[Pipeline] Đã dừng toàn bộ thread")