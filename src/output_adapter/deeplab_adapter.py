import cv2
import numpy as np

class DeeplabAdapter:
    def __init__(self):
        print("[*] Đã tải module DeepLabV3+")

    def infer(self, frame):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 1. Bóp hẹp vùng ROI lại để tập trung vào mặt đường, né hộ lan 2 bên
        mask_roi = np.zeros_like(edges)
        polygon = np.array([[
            (int(width * 0.15), height),
            (int(width * 0.85), height),
            (int(width * 0.55), int(height * 0.6)),
            (int(width * 0.45), int(height * 0.6))
        ]], np.int32)
        cv2.fillPoly(mask_roi, polygon, 255)

        masked_edges = cv2.bitwise_and(edges, mask_roi)
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 40, minLineLength=40, maxLineGap=150)

        # Mask đầu ra ban đầu là mảng đen
        lane_mask = np.zeros((height, width), dtype=np.uint8)

        left_x, left_y = [], []
        right_x, right_y = [], []

        # 2. Phân loại các đường tìm được thành vạch trái và vạch phải
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1: continue # Tránh lỗi chia cho 0
                slope = (y2 - y1) / (x2 - x1)
                
                # Lọc theo độ dốc (slope) và vị trí màn hình
                if slope < -0.3 and x1 < width // 2 and x2 < width // 2:
                    left_x.extend([x1, x2])
                    left_y.extend([y1, y2])
                elif slope > 0.3 and x1 > width // 2 and x2 > width // 2:
                    right_x.extend([x1, x2])
                    right_y.extend([y1, y2])

        # 3. Tạo hình thang an toàn (Dùng khi xe đi vào đoạn mất vạch kẻ đường)
        bottom_y = height
        top_y = int(height * 0.65)
        l_bottom_x = int(width * 0.25)
        l_top_x = int(width * 0.45)
        r_bottom_x = int(width * 0.75)
        r_top_x = int(width * 0.55)

        # Dùng hồi quy tuyến tính (polyfit) để tính toán ra 1 đường trung bình mượt mà nhất
        if len(left_x) > 0:
            poly_left = np.poly1d(np.polyfit(left_y, left_x, 1))
            l_bottom_x = int(poly_left(bottom_y))
            l_top_x = int(poly_left(top_y))

        if len(right_x) > 0:
            poly_right = np.poly1d(np.polyfit(right_y, right_x, 1))
            r_bottom_x = int(poly_right(bottom_y))
            r_top_x = int(poly_right(top_y))

        # 4. TÔ KÍN mặt đường để khớp với Logic Cảnh Báo
        lane_polygon = np.array([[
            (l_bottom_x, bottom_y),
            (l_top_x, top_y),
            (r_top_x, top_y),
            (r_bottom_x, bottom_y)
        ]], np.int32)

        cv2.fillPoly(lane_mask, lane_polygon, 255)

        return lane_mask
