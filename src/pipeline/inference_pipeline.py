import cv2
import torch
import numpy as np
from torchvision import transforms
# Đã thêm DeepLabV3_ResNet50_Weights vào thư viện import
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from ultralytics import YOLO

class ADASModelHandler:
    # Tôi đặt deeplab_weights_path=None để code cũ bên main.py của bạn không bị lỗi
    def __init__(self, yolo_weights_path, deeplab_weights_path=None, device=None):
        """
        Khởi tạo và load pre-trained weights cho cả 2 mô hình.
        Sử dụng weights mặc định của PyTorch cho DeepLabV3+ để test tự động tải.
        """
        # Ưu tiên sử dụng GPU (CUDA) để chạy real-time trên Windows
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[*] Đang khởi tạo luồng suy luận trên thiết bị: {self.device.upper()}")

        # 1. Load YOLOv8 (Object Detection)
        self.yolo_model = YOLO(yolo_weights_path)
        self.yolo_model.to(self.device)
        print("[+] Load thành công YOLOv8.")

        # 2. Load DeepLabV3+ (Lane Segmentation - Tự động tải từ PyTorch)
        print("[*] Đang nạp weights mặc định cho DeepLabV3+ (Sẽ mất chút thời gian tải ở lần chạy đầu tiên)...")
        self.deeplab_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self.deeplab_model.to(self.device)
        self.deeplab_model.eval() # Khóa các tham số để suy luận ổn định
        print("[+] Load thành công DeepLabV3+.")

        # 3. Định nghĩa tiền xử lý (Pre-processing) cho DeepLabV3+
        self.deeplab_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def process_frame(self, frame):
        """
        Đẩy frame ảnh qua 2 mô hình và trả về Bounding Boxes và Segmentation Mask.
        """
        original_h, original_w = frame.shape[:2]

        # ==========================================
        # WORKFLOW 1: OBJECT DETECTION (YOLOv8)
        # ==========================================
        yolo_results = self.yolo_model(frame, verbose=False)[0]
        
        detection_outputs = []
        
        # Mapping COCO classes to ADAS requirements:
        # 0: person (người đi bộ)
        # 2: car (xe hơi), 3: motorcycle (xe máy), 5: bus, 7: truck
        # 11: stop sign (biển báo)
        target_classes = {
            0: "Nguoi di bo",
            2: "Xe hoi",
            3: "Xe may",
            5: "Xe hoi (Bus)",
            7: "Xe hoi (Truck)",
            11: "Bien bao"
        }
        
        for box in yolo_results.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            if cls_id not in target_classes:
                continue # Skip unrequired classes
                
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            
            detection_outputs.append({
                "box": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "label": target_classes[cls_id]
            })

        # ==========================================
        # WORKFLOW 2: LANE SEGMENTATION (DeepLabV3+)
        # ==========================================
        # Tiền xử lý: Chuyển BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        input_tensor = self.deeplab_transform(rgb_frame).unsqueeze(0).to(self.device)

        # Chạy suy luận (Forward pass)
        with torch.no_grad():
            output = self.deeplab_model(input_tensor)['out'][0]
        
        # Hậu xử lý: Lấy index của class có xác suất cao nhất tại mỗi pixel
        segmentation_mask = torch.argmax(output, dim=0).cpu().numpy().astype(np.uint8)

        # ==========================================
        # WORKFLOW 2B: SIMULATED LANE SEGMENTATION
        # (Để đạt yêu cầu demo làn đường do COCO/VOC không có class "Lane")
        # ==========================================
        simulated_lane_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # ROI tập trung vào phần nửa dưới ảnh (mặt đường)
        roi_vertices = np.array([[(original_w * 0.1, original_h), 
                                  (original_w * 0.45, original_h * 0.6), 
                                  (original_w * 0.55, original_h * 0.6), 
                                  (original_w * 0.9, original_h)]], dtype=np.int32)
        roi_mask = np.zeros_like(edges)
        cv2.fillPoly(roi_mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        
        # Dilation để làm dày line thành lane mask
        kernel = np.ones((10, 10), np.uint8)
        dilated_lanes = cv2.dilate(masked_edges, kernel, iterations=2)
        
        # Gán label 1 cho lane
        simulated_lane_mask[dilated_lanes > 0] = 1
        segmentation_mask = simulated_lane_mask

        # Trả về đầu ra ổn định
        return detection_outputs, segmentation_mask