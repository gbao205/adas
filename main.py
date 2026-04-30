import cv2
import numpy as np
from src.pipeline.inference_pipeline import ADASModelHandler
from configs.config import VIDEO_PATH

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