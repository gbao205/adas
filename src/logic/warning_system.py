import cv2
from configs.config import ADASConfig

class WarningSystem:
    def __init__(self):
        self.config = ADASConfig()

    def check_collision(self, detected_objects, frame_height):
        """
        Kiểm tra va chạm dựa vào y_max của bounding box so với đáy màn hình.
        """
        collision_warning = False
        
        if detected_objects is None or len(detected_objects) == 0:
            return collision_warning

        for obj in detected_objects:
            # Ép kiểu về int đề phòng dữ liệu là tensor hoặc float
            x1, y1, x2, y2 = map(int, obj[:4])
            
            # Khoảng cách từ mép dưới bánh xe (y2) đến mui xe của mình (đáy frame)
            distance_to_bottom = frame_height - y2
            
            # Nếu xe phía trước lớn (chiếm nhiều diện tích) VÀ nằm quá gần đáy màn hình -> Cảnh báo
            # (Bạn có thể tinh chỉnh số 150 này trong file config)
            if distance_to_bottom < self.config.SAFE_DISTANCE_PIXELS:
                collision_warning = True
                break # Chỉ cần 1 xe chạm ngưỡng là hú còi, bỏ qua các xe khác

        return collision_warning

    def check_lane_departure(self, lane_mask, frame_width, frame_height):
        """
        Kiểm tra lấn làn bằng cách xét 'điểm neo' (anchor point) ở giữa mui xe.
        """
        if lane_mask is None:
            return False
            
        # 1. Đảm bảo mask có cùng kích thước với frame gốc (RẤT QUAN TRỌNG)
        # DeepLab thường trả ra mask nhỏ hơn (vd: 512x512), phải phóng to lên
        if lane_mask.shape[:2] != (frame_height, frame_width):
            lane_mask = cv2.resize(lane_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

        # 2. Tạo điểm neo ảo (Mũi xe của mình)
        anchor_x = frame_width // 2
        anchor_y = frame_height - 50 # Lùi lên 50 pixel từ đáy màn hình để tránh nhiễu
        
        # 3. Kiểm tra xem mũi xe có đang đè lên làn đường an toàn (mask > 0) không
        # Nếu giá trị pixel tại điểm neo là 0 -> Xe đã chệch ra ngoài làn được phân dải
        is_on_lane = lane_mask[anchor_y, anchor_x] > 0
        
        lane_warning = not is_on_lane
        return lane_warning