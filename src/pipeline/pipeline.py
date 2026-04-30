import queue
import time
import cv2
from configs.config import QUEUE_SIZE, NUM_WORKERS, VISUALIZER_MAX_SIZE, VISUALIZER_SKIP_FRAMES, SHOW_WINDOW
from src.video_input.video_reader import VideoReader
from src.pipeline.worker import Worker
from src.utils.fps import FPS
from src.utils.visualizer import Visualizer
from src.pipeline.perception import PerceptionAnalyzer
from src.pipeline.inference_pipeline import ADASModelHandler

# Evaluation Metrics: Import evaluation modules
from src.evaluation.benchmark import PerformanceBenchmark


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

        # Visualizer
        self.visualizer = Visualizer(
            max_display_size=VISUALIZER_MAX_SIZE,
            visualize_every_n_frames=VISUALIZER_SKIP_FRAMES
        )

        # Perception
        self.perception_analyzer = PerceptionAnalyzer(min_box_area=800)

        self.model_handler = ADASModelHandler(yolo_weights_path="yolov8n.pt")

        self.fps = FPS()
        self.running = False
        
        # Evaluation Metrics: Performance Benchmark
        self.benchmark = PerformanceBenchmark(window_size=100)
        print("[+] Đã khởi tạo Performance Benchmark")

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
        
        # Evaluation Metrics: Bắt đầu benchmark
        self.benchmark.start_benchmark()

        try:
            while True:
                try:
                    frame, yolo_img, deeplab_img = self.output_q.get(timeout=2)
                except queue.Empty:
                    if not self.reader.running:
                        break
                    continue

                # Evaluation Metrics: Bắt đầu đo frame
                self.benchmark.start_frame()

                # Inference với timing
                inference_start = time.time()
                detections, mask = self.model_handler.process_frame(frame)
                inference_time = (time.time() - inference_start) * 1000  # ms
                self.benchmark.record_inference_time(inference_time)
                
                # Perception với timing
                perception_start = time.time()
                valid_objects = self.perception_analyzer.process(detections, mask)
                perception_time = (time.time() - perception_start) * 1000  # ms
                self.benchmark.record_postprocessing_time(perception_time)
                
                warning_level = 0  # Placeholder: 0 = an toàn, >0 = cảnh báo

                if len(valid_objects) > 0:
                    # Tận dụng self.fps.count để làm ID cho Frame hiện tại
                    print(f"\n[Perception - Frame {self.fps.count}] Lọc thành công {len(valid_objects)} vật thể:")
                    for i, obj in enumerate(valid_objects):
                        cls_name = "Người" if obj['class_id'] == 0 else "Xe"
                        print(f"  -> {cls_name} {i+1}: Conf={obj['confidence']:.2f} | Vị trí={obj['position']} | Khoảng cách ảo={obj['distance_factor']:.2f}")

                # Tính FPS hiện tại
                elapsed = time.time() - start_time
                current_fps = self.fps.count / elapsed if elapsed > 0 else 0.0

                # Chuyển đổi định dạng valid_objects trả về cho Visualizer vẽ
                # Visualizer hiện tại đang mong đợi list dạng [x1, y1, x2, y2, conf, cls]
                display_detections = []
                for obj in valid_objects:
                    display_detections.append([*obj["box"], obj["confidence"], obj["class_id"]])

                # Vẽ lên màn hình
                display_frame = self.visualizer.draw_outputs(
                    frame, display_detections, mask, warning_level, fps_value=current_fps
                )

                if SHOW_WINDOW:
                    cv2.imshow("ADAS Pipeline", display_frame)
                    # Nhấn 'q' để thoát, chờ 1ms để xử lý sự kiện window
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[Pipeline] Nhận lệnh thoát từ ngườdùng (phím 'q')")
                        break

                self.fps.increment()
                
                # Evaluation Metrics: Kết thúc đo frame
                self.benchmark.end_frame()
                self.benchmark.record_gpu_memory()

                # In thông tin mỗi 30 frame
                if self.fps.count % 30 == 0:
                    # Evaluation Metrics: Lấy metrics từ benchmark
                    benchmark_fps = self.benchmark.get_current_fps()
                    latency = self.benchmark.get_latency_stats()
                    
                    print(f"[INFO] FPS: {current_fps:.2f} (Benchmark: {benchmark_fps:.2f})")
                    print(f"[INFO] Latency: {latency['mean']:.2f}ms (P95: {latency['p95']:.2f}ms)")
                    print(f"[INFO] FrameQ: {self.frame_q.qsize()} | OutputQ: {self.output_q.qsize()}")

                self.output_q.task_done()

        except KeyboardInterrupt:
            print("[Pipeline] Bị dừng bởi ngườdùng")

        finally:
            if SHOW_WINDOW:
                cv2.destroyAllWindows()
            self.stop()

            print("\n===== SUMMARY =====")
            print(f"Frames processed: {self.fps.count}")
            print(f"Average FPS: {self.fps.get():.2f}")
            print("===================")
            
            # Evaluation Metrics: In báo cáo benchmark chi tiết
            print("\n")
            self.benchmark.print_summary()
            
            # Evaluation Metrics: Kiểm tra real-time
            realtime_status = self.benchmark.get_realtime_status(target_fps=30.0)
            print(f"\n{realtime_status}")

    def stop(self):
        print("[Pipeline] Dừng pipeline...")
        self.running = False
        self.reader.running = False
        for w in self.workers:
            w.running = False
            
        # Giải phóng các thread đang bị kẹt (block) trong Queue
        # Bằng cách rút cạn (drain) các hàng đợi
        while not self.frame_q.empty():
            try:
                self.frame_q.get_nowait()
                self.frame_q.task_done()
            except queue.Empty:
                break
                
        while not self.output_q.empty():
            try:
                self.output_q.get_nowait()
                self.output_q.task_done()
            except queue.Empty:
                break

        # Đợi thread kết thúc
        self.reader.join()
        for w in self.workers:
            w.join(timeout=2) # Evaluation Metrics timeout 
            
        print("[Pipeline] Đóng toàn bộ thread thành công")

