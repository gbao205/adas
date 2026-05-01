from ultralytics import YOLO
from configs.config import ADASConfig

class YoloAdapter:
    def __init__(self):
        self.config = ADASConfig()
        # Khởi tạo model YOLOv8 (Tự động tải yolov8n.pt nếu chưa có trong thư mục)
        self.model = YOLO(self.config.YOLO_WEIGHTS)

    def infer(self, frame):
        """
        Chạy inference YOLOv8 và trả về list các box format: [x1, y1, x2, y2, conf, class_id]
        """
        # Chạy model trên frame (verbose=False để terminal không bị spam log)
        results = self.model(frame, verbose=False, half=True)
        boxes_output = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Chỉ lọc lấy các class xe cộ: 0(Person), 2(Car), 3(Motorcycle), 5(Bus), 7(Truck)
                valid_classes = [0, 2, 3, 5, 7]
                
                if conf > self.config.CONFIDENCE_THRESHOLD and cls_id in valid_classes:
                    # Tách tọa độ x_min, y_min, x_max, y_max
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes_output.append([int(x1), int(y1), int(x2), int(y2), conf, cls_id])
                    
        return boxes_output