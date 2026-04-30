import cv2
import numpy as np
from src.pipeline.inference_pipeline import ADASModelHandler
from configs.config import VIDEO_PATH

def perception_fusion(detections, lane_mask):
    """
    Ráp tọa độ Object Detection lên mặt nạ Lane Segmentation.
    Xác định xem vật thể có đang nằm trong làn đường hay không.
    """
    fused_results = []
    height, width = lane_mask.shape

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        
        # Phép toán hình học: Xác định (anchor point) của xe.
        # Thường lấy điểm giữa của cạnh dưới bounding box (vị trí bánh xe chạm đường).
        center_x = int((x1 + x2) / 2)
        bottom_y = int(y2)
        
        # Đảm bảo tọa độ không vượt quá kích thước frame để tránh lỗi Index Out of Bounds
        center_x = max(0, min(center_x, width - 1))
        bottom_y = max(0, min(bottom_y, height - 1))
        
        # Kiểm tra điểm neo có nằm trên mask phân vùng làn đường (class_id = 1) hay không
        in_lane = False
        if lane_mask[bottom_y, center_x] == 1:
            in_lane = True
            
        fused_results.append({
            "box": det["box"],
            "class_id": det["class_id"],
            "confidence": det["confidence"],
            "in_lane": in_lane,
            "anchor_point": (center_x, bottom_y)
        })
        
    return fused_results

def certification_check(fused_objects):
    """
    Lọc nhiễu và kiểm định logic (loại bỏ các frame bị nhiễu do rung lắc hoặc vật thể rác).
    """
    valid_objects = []
    
    for obj in fused_objects:
        x1, y1, x2, y2 = obj["box"]
        
        # 1. Lọc nhiễu theo diện tích (loại bỏ các box quá nhỏ, có thể là nhiễu)
        box_area = (x2 - x1) * (y2 - y1)
        if box_area < 500: # Ngưỡng diện tích (có thể tùy chỉnh)
            continue
            
        # 2. Lọc nhiễu theo tỷ lệ khung hình (Aspect Ratio)
        # Xe cộ hoặc người thường có tỷ lệ chiều rộng / chiều cao nhất định
        aspect_ratio = (x2 - x1) / float(y2 - y1 + 1e-6)
        if aspect_ratio > 5.0 or aspect_ratio < 0.2:
            continue
            
        valid_objects.append(obj)
        
    return valid_objects

def main():
    # Khởi tạo pipeline với đường dẫn weights của bạn
    # Lưu ý: Cần chuẩn bị sẵn 2 file weights này trong folder dự án
    pipeline = ADASModelHandler(
        yolo_weights_path="yolov8n.pt",  
    )

    cap = cv2.VideoCapture("data/samples/test1.mp4") # Hoặc để 0 nếu dùng webcam

    if not cap.isOpened():
        print(f"[!] Lỗi: Không thể mở video. Hãy kiểm tra lại xem file đã tồn tại ở đường dẫn '{VIDEO_PATH}' chưa!")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Đẩy frame vào pipeline để lấy output
        detections, lane_mask = pipeline.process_frame(frame)

        # ==== XỬ LÝ HIỂN THỊ TRỰC QUAN ====
        # 1. Vẽ Bounding Boxes từ YOLO
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {det['class_id']} {det['confidence']:.2f}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2. Phủ (Overlay) Segmentation Mask từ DeepLabV3+
        # Giả sử class_id của làn đường là 1
        lane_pixels = (lane_mask == 1)
        
        # Tạo một lớp màu xanh dương (Blue) cho làn đường
        color_mask = np.zeros_like(frame)
        color_mask[lane_pixels] = (255, 0, 0) 
        
        # Trộn lớp màu mặt nạ với frame gốc
        frame = cv2.addWeighted(frame, 1, color_mask, 0.5, 0)

        cv2.imshow("ADAS Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()