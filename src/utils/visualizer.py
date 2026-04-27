import cv2
import numpy as np
import time

class Visualizer:
    """
    Trực quan hóa dữ liệu: vẽ box, tô màu làn đường, hiển thị text cảnh báo.
    Được tối ưu để giảm thiểu tác động đến FPS tổng thể.
    """
    def __init__(self, max_display_size=None, visualize_every_n_frames=1):
        """
        Args:
            max_display_size: Tuple (width, height) hoặc None. 
                              Nếu frame lớn hơn, sẽ resize trước khi vẽ.
            visualize_every_n_frames: Chỉ vẽ 1 frame sau mỗi N frame để giảm tải.
        """
        # Định nghĩa màu sắc (BGR)
        self.COLOR_LANE = (0, 255, 0)      # Xanh lá cho làn đường an toàn
        self.COLOR_WARNING = (0, 0, 255)   # Đỏ cho cảnh báo nguy hiểm
        self.COLOR_BOX = (255, 0, 0)       # Xanh dương cho vật thể
        self.COLOR_FPS = (255, 255, 0)     # Vàng cho FPS
        self.COLOR_TEXT_BG = (0, 0, 0)     # Đen cho nền text

        self.max_display_size = max_display_size
        self.visualize_every_n_frames = max(1, visualize_every_n_frames)
        self._frame_counter = 0

        # Cache cho warning text (pre-render)
        self._warning_text = "!!! WARNING: DANGER !!!"
        self._warning_font = cv2.FONT_HERSHEY_SIMPLEX
        self._warning_scale = 1.0
        self._warning_thickness = 3
        self._warning_size = cv2.getTextSize(
            self._warning_text, self._warning_font, 
            self._warning_scale, self._warning_thickness
        )[0]

        # Cache cho lane overlay (tái sử dụng buffer)
        self._lane_overlay = None
        self._lane_mask_prev = None

    def draw_outputs(self, frame, detections, mask, warning_level, fps_value=None):
        """
        Tổng hợp vẽ tất cả các lớp dữ liệu lên frame gốc.
        
        Args:
            frame: Ảnh gốc (BGR)
            detections: List các detection [x1, y1, x2, y2, conf, cls]
            mask: Mask segmentation (2D array)
            warning_level: 0 = an toàn, >0 = cảnh báo
            fps_value: Giá trị FPS để hiển thị (optional)
        
        Returns:
            Frame đã được vẽ (có thể là frame gốc nếu skip)
        """
        self._frame_counter += 1

        # Giảm tải: skip frame nếu cần
        if self._frame_counter % self.visualize_every_n_frames != 0:
            return frame

        # Giảm tải: resize frame nếu quá lớn
        original_shape = frame.shape
        frame = self._maybe_resize(frame)
        scale_x = frame.shape[1] / original_shape[1]
        scale_y = frame.shape[0] / original_shape[0]

        # 1. Vẽ mặt nạ làn đường (Segmentation)
        if mask is not None:
            # Resize mask nếu frame đã được resize
            if scale_x != 1.0 or scale_y != 1.0:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
            frame = self._draw_lane(frame, mask, warning_level)

        # 2. Vẽ khung bao vật thể (Object Detection)
        if detections is not None:
            frame = self._draw_boxes(frame, detections, scale_x, scale_y)

        # 3. Hiển thị cảnh báo chớp nháy (Realtime Warning)
        if warning_level > 0:
            frame = self._draw_flash_warning(frame)

        # 4. Hiển thị FPS
        if fps_value is not None:
            frame = self._draw_fps(frame, fps_value)

        return frame

    def _maybe_resize(self, frame):
        """Resize frame nếu vượt quá max_display_size để giảm tải vẽ."""
        if self.max_display_size is None:
            return frame
        
        max_w, max_h = self.max_display_size
        h, w = frame.shape[:2]
        
        if w <= max_w and h <= max_h:
            return frame
        
        # Tính scale giữ tỉ lệ
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _draw_lane(self, frame, mask, warning_level):
        """
        Tối ưu việc tô màu làn đường.
        Dùng bitwise operations thay vì copy toàn bộ frame + addWeighted.
        """
        color = self.COLOR_LANE if warning_level == 0 else self.COLOR_WARNING
        
        # Tạo overlay màu chỉ cho vùng mask (không copy toàn bộ frame)
        h, w = frame.shape[:2]
        
        # Khởi tạo overlay cache nếu kích thước thay đổi
        if self._lane_overlay is None or self._lane_overlay.shape[:2] != (h, w):
            self._lane_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            self._lane_overlay.fill(0)  # Reset về 0 thay vì tạo mới
        
        # Tô màu chỉ vùng mask
        self._lane_overlay[mask > 0] = color
        
        # Blend: frame = frame * 0.6 + overlay * 0.4
        # Dùng cv2.addWeighted trực tiếp (nhanh hơn manual blend)
        cv2.addWeighted(self._lane_overlay, 0.4, frame, 0.6, 0, dst=frame)
        
        return frame

    def _draw_boxes(self, frame, detections, scale_x=1.0, scale_y=1.0):
        """
        Vẽ bounding boxes từ kết quả của YOLOv8.
        Tối ưu: giảm số lần ép kiểu, dùng LINE_8.
        """
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            
            # Scale tọa độ nếu frame đã resize
            x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
            x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
            
            label = f"V:{conf:.2f}"
            
            # Vẽ box với LINE_8 (nhanh hơn LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_BOX, 2, cv2.LINE_8)
            
            # Vẽ label với nền đen để dễ đọc
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), self.COLOR_BOX, -1)
            cv2.putText(frame, label, (x1, y1 - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_8)
        
        return frame

    def _draw_flash_warning(self, frame):
        """
        Tạo hiệu ứng chớp nháy dựa trên timestamp thực.
        Ưu điểm: không phụ thuộc vào frame_count, hiệu ứng đều đặn dù skip frame.
        """
        # Nhấp nháy 2 lần/giây (chu kỳ 500ms)
        if int(time.time() * 2) % 2 == 0:
            h, w = frame.shape[:2]
            tx = (w - self._warning_size[0]) // 2
            ty = 50
            
            # Vẽ nền đỏ cho text
            padding = 10
            cv2.rectangle(
                frame, 
                (tx - padding, ty - self._warning_size[1] - padding),
                (tx + self._warning_size[0] + padding, ty + padding),
                self.COLOR_WARNING, -1
            )
            
            # Vẽ text trắng
            cv2.putText(
                frame, self._warning_text, (tx, ty),
                self._warning_font, self._warning_scale, 
                (255, 255, 255), self._warning_thickness, cv2.LINE_8
            )
        
        return frame

    def _draw_fps(self, frame, fps_value):
        """Hiển thị FPS ở góc trái dưới."""
        text = f"FPS: {fps_value:.1f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        h, w = frame.shape[:2]
        
        # Nền đen mờ
        cv2.rectangle(frame, (10, h - th - 20), (10 + tw + 10, h - 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_FPS, 2, cv2.LINE_8)
        
        return frame

