import cv2
import time
from src.video_input.video_reader import VideoReader
from src.output_adapter.yolo_adapter import YoloAdapter
from src.output_adapter.deeplab_adapter import DeeplabAdapter
from src.logic.warning_system import WarningSystem
from src.utils.visualizer import Visualizer
from src.pipeline.perception import PerceptionAnalyzer


def main():
    video_path = "data/test4.mp4"
    output_video_path = "demo_result.mp4"
    output_report_path = "metrics_report.txt"

    stream = cv2.VideoCapture(video_path)
    yolo = YoloAdapter()
    deeplab = DeeplabAdapter()
    warning_sys = WarningSystem()
    vis = Visualizer()

    # Khởi tạo PerceptionAnalyzer (Perception Fusion + Certification Check)
    perception = PerceptionAnalyzer(lane_class_id=0, min_box_area=800)

    frame_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(stream.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (frame_width, frame_height))

    print("Bắt đầu xử lý luồng ADAS và đo lường chỉ số...")

    # Biến để đo FPS
    total_frames = 0
    start_time = time.time()

    while stream.isOpened():
        ret, frame = stream.read()
        if not ret:
            break

        total_frames += 1

        # 1. AI Inference
        detections_raw = yolo.infer(frame)  # Raw detections: [x1, y1, x2, y2, conf, cls_id]
        lane_mask = deeplab.infer(frame)

        # 2. Perception: Fusion + Certification Check (Đỗ Huỳnh Đức)
        # Chuyển đổi format detections từ list sang dict cho PerceptionAnalyzer
        detections_dict = []
        for det in detections_raw:
            detections_dict.append({
                "box": [det[0], det[1], det[2], det[3]],  # [x1, y1, x2, y2]
                "class_id": det[5],  # class_id
                "confidence": det[4]  # confidence
            })

        # Chạy Perception Fusion và Certification Check
        valid_objects = perception.process(detections_dict, lane_mask)

        # Chuyển đổi valid_objects trở lại format [x1, y1, x2, y2, conf, cls_id] cho Visualizer
        objects_filtered = []
        for obj in valid_objects:
            x1, y1, x2, y2 = obj["box"]
            conf = obj["confidence"]
            cls_id = obj["class_id"]
            objects_filtered.append([x1, y1, x2, y2, conf, cls_id])

        # 3. Logic (Warning System) - dùng kết quả đã lọc từ Perception
        collision_alert = warning_sys.check_collision(objects_filtered, frame_height)
        lane_alert = warning_sys.check_lane_departure(lane_mask, frame_width, frame_height)

        # 4. Tăng cường cảnh báo dựa trên Perception (IN_LANE = nguy hiểm hơn)
        # Nếu có object nào ở trong làn (IN_LANE), tăng mức cảnh báo
        perception_warning = 0
        for obj in valid_objects:
            if obj.get("position") == "IN_LANE":
                perception_warning = 2  # Nguy hiểm cao - đang đi trên làn đường
            elif obj.get("position") == "CROSSING" and perception_warning < 1:
                perception_warning = 1  # Cảnh báo trung bình - đang băng qua
        # Kết hợp với collision_alert từ WarningSystem
        if collision_alert:
            perception_warning = max(perception_warning, 1)

        # 5. Visualizer
        final_frame = vis.draw_lane_and_boxes(
            frame=frame,
            objects=objects_filtered,  # D��ng objects đã lọc từ Perception
            lane_mask=lane_mask,
            collision_warning=collision_alert,
            lane_warning=lane_alert
        )

        # (Tùy chọn) Vẽ thêm thông tin Perception lên frame debug
        # final_frame = vis.draw_perception_info(final_frame, valid_objects)

        out.write(final_frame)

        cv2.imshow("Leader Check", cv2.resize(final_frame, (1600, 900)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Tính toán FPS trung bình
    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = total_frames / total_time if total_time > 0 else 0

    stream.release()
    out.release()
    cv2.destroyAllWindows()

    # --- XUẤT BÁO CÁO CHỈ SỐ (TESTING & EVALUATION) ---
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("      BÁO CÁO ĐÁNH GIÁ HỆ THỐNG ADAS    \n")
        f.write("==================================================\n\n")
        f.write(f"- Video đầu vào: {video_path}\n")
        f.write(f"- Tổng số frame xử lý: {total_frames}\n")
        f.write(f"- Thời gian xử lý: {total_time:.2f} giây\n\n")

        f.write("1. CHỈ SỐ HIỆU NĂNG (Thực tế đo lường):\n")
        f.write(f"   => FPS Trung bình: {avg_fps:.2f} frames/sec\n\n")

        f.write("2. CHỈ SỐ ĐỘ CHÍNH XÁC (Dựa trên Model Architecture):\n")
        f.write("   => mAP (YOLOv8n - Object Detection): ~37.3%\n")
        f.write("   => mIoU (DeepLabV3+ - Lane Seg): ~70.5%\n")
        f.write("==================================================\n")

    print(f"Hoàn tất! Đã lưu video: {output_video_path}")
    print(f"Đã xuất báo cáo chỉ số tại: {output_report_path}")


if __name__ == "__main__":
    main()
