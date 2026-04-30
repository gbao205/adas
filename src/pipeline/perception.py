import numpy as np

class PerceptionAnalyzer:
    """
    Module phụ trách bởi Đỗ Huỳnh Đức.
    Nhiệm vụ: Perception Fusion (Ráp tọa độ) và Certification Check (Lọc nhiễu).
    """
    def __init__(self, lane_class_id=1, min_box_area=800):
        self.lane_class_id = lane_class_id
        self.min_box_area = min_box_area

    def process(self, detections, lane_mask):
        """
        Hàm thực thi toàn bộ luồng của Đỗ Huỳnh Đức.
        """
        if not detections or lane_mask is None:
            return []

        # 1. Perception Fusion
        fused_data = self._fusion(detections, lane_mask)
        
        # 2. Certification Check
        valid_data = self._certification_check(fused_data)
        
        return valid_data

    def _fusion(self, detections, lane_mask):
        """
        Tính toán phần trăm diện tích vật thể nằm trên làn đường.
        """
        fused_results = []
        height, width = lane_mask.shape
        binary_lane_mask = (lane_mask == self.lane_class_id)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            
            # Cắt frame tránh lỗi Index Out of Bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            
            box_area = (x2 - x1) * (y2 - y1)
            if box_area == 0:
                continue
                
            # Trích xuất vùng mask tương ứng với box
            box_mask_region = binary_lane_mask[y1:y2, x1:x2]
            overlap_area = np.sum(box_mask_region)
            overlap_ratio = overlap_area / float(box_area)
            
            # Phân loại vị trí
            if overlap_ratio > 0.6:
                position = "IN_LANE"
            elif overlap_ratio >= 0.1:
                position = "CROSSING"
            else:
                position = "OUT"
                
            # Hệ số khoảng cách ảo (0 -> 1, càng gần 1 càng sát xe)
            distance_factor = y2 / float(height) 
                
            fused_results.append({
                "box": [x1, y1, x2, y2],
                "class_id": det["class_id"],
                "confidence": det["confidence"],
                "overlap_ratio": overlap_ratio,
                "position": position,
                "distance_factor": distance_factor
            })
            
        return fused_results

    def _certification_check(self, fused_objects):
        """
        Lọc nhiễu hình học, rung lắc và loại bỏ các vùng chết (dead zones) của camera.
        """
        valid_objects = []
        
        for obj in fused_objects:
            x1, y1, x2, y2 = obj["box"]
            cls_id = obj["class_id"]
            conf = obj["confidence"]
            dist_factor = obj["distance_factor"] # Tỷ lệ y2 / height (từ 0.0 đến 1.0)
            
            # 1. LỌC THEO ĐỘ TỰ TIN (Ngăn báo động giả)
            # Loại bỏ các nhận diện lờ mờ (như bóng râm, phản chiếu trên kính)
            if conf < 0.40:
                continue

            w = x2 - x1
            h = y2 - y1
            box_area = w * h
            
            # 2. LỌC THEO DIỆN TÍCH
            if box_area < self.min_box_area:
                continue
                
            # 3. LỌC THEO TỶ LỆ KHUNG HÌNH (Aspect Ratio)
            aspect_ratio = w / float(h + 1e-6)
            
            if cls_id == 0 and aspect_ratio > 1.5: 
                continue # Người đi bộ nhưng lại là hình chữ nhật nằm ngang (nhiễu)
                
            if cls_id in [2, 3, 5, 7] and (aspect_ratio < 0.5 or aspect_ratio > 4.5):
                continue # Xe cộ nhưng box quá hẹp dọc hoặc quá dẹt (nhiễu)
                
            # 4. LỌC THEO KHÔNG GIAN (Đỉnh và Đáy camera)
            # Tránh nhận diện mây/chim chóc trên trời
            if dist_factor < 0.3:
                continue 
            
            # TRÁNH VÙNG CHẾT (Dead Zone) CỦA XE
            # Bất kỳ vật thể nào có mép dưới (y2) vọt qua mốc 85% chiều cao khung hình
            # thì 99% đó là nắp capo, taplo, hoặc camera hành trình dán trên kính
            if dist_factor > 0.85:
                continue
                
            valid_objects.append(obj)
            
        return valid_objects