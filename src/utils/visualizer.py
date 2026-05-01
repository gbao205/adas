import cv2
import numpy as np
from datetime import datetime

class Visualizer:
    def __init__(self):
        # Cấu hình màu sắc
        self.color_box_normal = (0, 255, 0)
        self.color_box_warning = (0, 0, 255)
        self.color_lane_normal = (255, 0, 0)
        self.color_lane_warning = (0, 0, 255)
        self.alpha = 0.4
        
        # Bộ đếm frame để làm hiệu ứng nhấp nháy và giả lập tốc độ
        self.frame_count = 0

    def draw_hud_panel(self, frame, text, position, bg_color, text_color):
        """Hàm vẽ bảng thông tin (HUD) mờ đằng sau text"""
        x, y = position
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        overlay = frame.copy()
        # Vẽ khối chữ nhật làm nền
        cv2.rectangle(overlay, (x - 10, y - h - 10), (x + w + 10, y + 10), bg_color, -1)
        # Trộn nền để tạo độ trong suốt 50%
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Ghi chữ lên trên
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        return frame

    def draw_lane_and_boxes(self, frame, objects, lane_mask, collision_warning=False, lane_warning=False):
        self.frame_count += 1
        height, width = frame.shape[:2]

        # 1. OVERLAY SEGMENTATION MASK
        if lane_mask is not None:
            overlay = frame.copy()
            lane_color = self.color_lane_warning if lane_warning else self.color_lane_normal
            mask_colored = np.zeros_like(frame)
            mask_colored[lane_mask > 0] = lane_color
            idx = (lane_mask > 0)
            overlay[idx] = mask_colored[idx]
            frame = cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0)

        # 2. VẼ BOUNDING BOXES & GIẢ LẬP KHOẢNG CÁCH
        if objects is not None:
            # Từ điển ánh xạ class_id sang tên gọi
            class_names = {0: "Pedestrian", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
            
            for obj in objects:
                x1, y1, x2, y2 = map(int, obj[:4])
                
                # Lấy class_id (nằm ở vị trí thứ 5 trong mảng YOLO trả về)
                cls_id = int(obj[5]) if len(obj) > 5 else 2
                obj_name = class_names.get(cls_id, "Vehicle") # Nếu không có trong dict thì để mặc định là Vehicle
                
                box_color = self.color_box_warning if collision_warning else self.color_box_normal
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Giả lập tính toán khoảng cách mét (dựa vào tọa độ y của xe)
                distance_sim = max(5.0, round((height - y2) * 0.15, 1))
                
                if collision_warning:
                    label = f"!!! BRAKE: {distance_sim}m !!!"
                else:
                    # Thay thế chữ "Car" cứng nhắc bằng tên thật của object
                    label = f"{obj_name}: {distance_sim}m"
                    
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # 3. VẼ GIAO DIỆN HUD TỔNG THỂ
        # a. Tốc độ và giờ hệ thống (Góc dưới)
        time_str = datetime.now().strftime("%H:%M:%S")
        sys_info = f"ADAS SYS: ACTIVE | {time_str}"
        frame = self.draw_hud_panel(frame, sys_info, (20, height - 30), (0, 0, 0), (0, 255, 255))

        # Tốc độ thay đổi chút ít cho giống xe đang chạy
        speed_sim = 68 + (self.frame_count % 3) 
        speed_text = f"SPEED: {speed_sim} km/h"
        frame = self.draw_hud_panel(frame, speed_text, (width - 250, height - 30), (0, 0, 0), (255, 255, 255))

        # b. Trạng thái các Module (Góc trên)
        lane_text = "LANE TRACKING: ALERT" if lane_warning else "LANE TRACKING: OK"
        lane_color = (0, 0, 255) if lane_warning else (0, 255, 0)
        frame = self.draw_hud_panel(frame, lane_text, (20, 40), (0, 0, 0), lane_color)

        fcw_text = "FCW RADAR: ALERT" if collision_warning else "FCW RADAR: CLEAR"
        fcw_color = (0, 0, 255) if collision_warning else (0, 255, 0)
        frame = self.draw_hud_panel(frame, fcw_text, (20, 90), (0, 0, 0), fcw_color)

        # 4. TÍN HIỆU CẢNH BÁO KHẨN CẤP NHẤP NHÁY
        if collision_warning or lane_warning:
            # Thuật toán nhấp nháy: chia dư cho 10, nếu bé hơn 5 thì nháy (tốc độ chớp vừa phải)
            if self.frame_count % 10 < 5:
                # Vẽ viền đỏ bao quanh toàn bộ kính lái
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 12)
                
                # In dòng chữ lớn khẩn cấp giữa màn hình
                alert_msg = "!!! COLLISION IMMINENT !!!" if collision_warning else "!!! LANE DEPARTURE !!!"
                
                # Căn giữa chữ
                text_size = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
                text_x = (width - text_size[0]) // 2
                text_y = int(height * 0.3)
                
                # Đổ bóng đen cho chữ dễ đọc trên nền sáng
                cv2.putText(frame, alert_msg, (text_x+2, text_y+2), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 4)
                cv2.putText(frame, alert_msg, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

        return frame